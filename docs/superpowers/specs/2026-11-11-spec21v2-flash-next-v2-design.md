# Spec 21 v2 - Qwen3.8-Flash-Next whole on two B70s: our own weights, expert tiering, the second card

**Status:** design, 2026-11-11; **revised 2026-11-12** with the operator's answers to the first draft's questions
(§0) and what the other engines on this model do (Strata, XeStrata, saber, basalt: read from their repos, nothing of
theirs run here). It continues spec 21 (the `qwen4_exp` family, built) and spec 22 (the expert-offload tier: its
probe tools built, the tier not built), which stay as the historical record.
**Plans:** [21v2a](../plans/2026-11-11-spec21v2a-whole-model-tiering.md) (our quantisation, the probe, the tier on
`--parallel pp`), then [21v2b](../plans/2026-11-12-spec21v2b-tensor-parallel.md) (tensor parallel with the expert
split).

Every number is **measured** unless marked **derived** (from the checkpoint headers and the layout header),
**estimated**, **published** (another engine's own report) or **proposed** (a bar for the operator to approve).

## 0. The operator's decisions (2026-11-12)

1. **Our own weights.** Flash-Next is re-quantised by us with AutoRound int4 g64 symmetric (the one format) from the
   bf16 original `Qwen/Qwen3.8-Flash-Next` (360 GB; the operator downloads it to the cold disk `~/b70/hdd/hf-hub` and
   links it into the HF cache), **the output head included** (`--quant_lm_head`) and **the MTP head's experts
   included**. AutoRound's low-CPU-memory mode reads one block at a time, so the cold disk is fast enough. Intel's
   g128 checkpoint stays the comparison and the development vehicle until ours passes (§2). This reverses the
   2026-11-07 drop of plan 21q.
2. **Pipeline parallel (the layer split) first, tensor parallel after.**
3. **Gates to start:** decode ≥ 30 t/s with `pp`, ≥ 50 t/s with `tp`, prefill ≥ 1000 t/s at 4k with `tp` (§11). **The
   goal is to beat Strata** (two B70s, 66 t/s on IQ3_XXS) at our better 4-bit quality. Take the best of Strata,
   XeStrata, saber and basalt; **P2P is mandatory wherever it helps**.
4. **The MTP head's experts** are quantised with the model; acceptance is measured (§7.2).
5. **Host RAM:** 128 GB in the box, **~115 GB usable** by our jobs (5 GB kept for the system).

Not decided (§12 question 1): the base format stays int4 g64 everywhere and mixed 3/4-bit experts (Strata's way to
fit two cards) are only a later lever if the gates are missed - the **recommended** default, awaiting the operator.

## 1. Where Flash-Next stands today (2026-11-12)

**Built and on main** (spec 21 §12-§16, spec 22a):
- the CPU reference (`tools/oracle/qwen4exp_ref.py`, transformers 5.19.0 layer-streamed, the MTP head from vLLM), the
  descriptor and loader (both forms: ours - int4 g64 dense, shared and routed - and Intel's - bf16 dense and shared,
  g128 experts through the exact g64 expansion; the int8 PLE table pinned in host memory in 16 ranges), decode,
  prefill and `b70-serve` with the prefix cache and the MTP head - all **truncated**: `--layers N` (on Intel's form 18
  of 48 layers fit one card, 37 fit two with `--parallel pp`). Our form has run only on synthetic real-width
  checkpoints (`tools/quantize/qwen4exp/make_synth.py`);
- spec 22a's probe tools: `tools/probe/probe_offload` (copy rates, zero-copy expert reads, replay visibility, host
  bandwidth, pinned capacity, the alias check), `tools/probe/offload_links.sh`, `tools/oracle/offload_curve.py`
  (hit-rate curves, swap simulation, MTP verify unions, prefill bytes, the routing profile, the projection) and the
  planner's routed-expert budget (`runtime::qwen4exp::expert_budget`).

**Passed on the cards (Intel's checkpoint):** `r31.load`, `r32.k3` / `golden` / `pp` / `cli`, `r33.prefill` / `pp`
(since 2026-11-07); **prefill against an f32 reference** on 18 layers (one card) and 37 (two cards) by the operator's
count rule - the prefill's rejected expert routes at most 1.2x decode's, its f32-accepted tokens at least decode's
(18 layers: ratio 1.17, tokens 32 / 32 both; 37 layers: ratio 1.06, tokens 29 / 32 prefill vs 28 / 32 decode;
[record](../../probes/q4exp-f32-aligned-routes-2026-11-10.md), merged 2026-11-10).

**Recorded:** teacher-forced routing traces on Intel's checkpoint, 39 sources, 474 MB (2026-11-09, 8095 s on the box
CPU, peak RSS 94.5 GiB) - the input of the hit-rate curve. The routers are bf16 in both checkpoints, so these traces
serve our checkpoint too until its own short trace confirms the profile (plan 21v2a Task 7). Nothing of
`probe_offload` has run on a card.

**Not passed or not run:** the golden-set remake with block scores (`r30.golden` / `r30.golden_layers`, held in the
CPU queue), and the two tests that wait for it (`r32.partial`, `r33.golden_real`); `r31.load_real`; the MTP CPU
stages `r34.fixture` / `r34.accept` / `r34.a4_ref`; `r34.a4` (needs the whole model); the PLE read rate
(`r32.ple_rate`); every speed row (none exists for Flash-Next).

**Open quality finding:** decode itself already disagrees with the f32 reference beyond the near-tie window on some
routes, from layer 0 (18 layers: 1089 of 81,576 layer-position routes rejected; 37 layers: 8413 of 167,684), and 4 of
32 decode tokens at 37 layers are not f32-accepted. The short diagnostic traces the first difference to bf16
activation rounding in layer 0's convolution, amplified where a route flips
([record](../../probes/q4exp-f32-decode-diagnostic-2026-11-10.md)); not attributed further. §10's twin count (approved
2026-11-12) turns it into a pass or a fail.

**What changed since spec 21 / 22 were written:**
1. **Weights:** Intel's checkpoint (bf16 dense, ~9.3 GB a token) was the weights from 2026-11-07; from 2026-11-12 our
   own AutoRound export is (§0.1, §2), with int4 dense layers and head: ~4.7 GB a token (§3).
2. **Spec 22 was dropped on 2026-11-06** (host-RAM offload on PCIe 3.0) and came back as a probe on 2026-11-07
   (open-work item 4) after the other engines' tiering designs.
3. **The two-card modes exist:** `--parallel tp` (tensor parallel over P2P, Qwen3.8 only; an all-reduce costs
   3.5 µs inside replayed lists, measured), `replica` (a model copy per card + slots, `--max-num-seqs`) and `pp`
   (layers split; the one Flash-Next uses today).
4. **Host memory both cards map stays at or below 4 GiB per allocation**: a 32 GiB pinned allocation imported by the
   second card made the kernel answer "busy" and the driver retry forever (the 2026-11-11 prefix cache hang); the
   prefix cache is now 2 GiB pinned pieces.
5. **Zero-config engine:** no tuning flags; the engine sizes and chooses. Spec 22's planned `--expert-*` flags are not
   built. Environment variables are allowed for experiments and tests only.
6. **Fixed slots, no paged KV pool** (operator ruling) - the slot design of `--max-num-seqs` applies.
7. **The quality bar is the f32 judge** (§10), not cosine bars against a bf16 reference.
8. **Removed 2026-11-07 (the never-run triage):** expert pruning - spec 22's P0.8, §4b's coding mask, plan 22m and their
   tools. They do not come back here (one format, best quality).

## 2. Our weights: the AutoRound re-quantisation

**What is quantised** (spec 21 §5 "Ours", with the head and the MTP experts now in): int4 g64 sym, export
`auto_round:auto_gptq` - routed and shared experts, QSA `q/k/v/o_proj`, GDN `in_proj_qkv` / `in_proj_z` /
`out_proj`, **`lm_head`** (`--quant_lm_head`), **the MTP head's 512 routed experts**. Kept bf16: routers,
`shared_expert_gate`, `in_proj_a/b`, the indexer, hyper-connections, every norm, the embedding, the MTP head's dense
projections and shared expert (`loader::q4_mtp_desc`), vision. **`ple` is in the ignore list**; the PLE table is
converted by `tools/quantize/qwen4exp/ple_int8.py`, and since AutoRound never touches it, the int8 file already made
from Intel's snapshot is reused if the original's table hashes equal (checked, not assumed).

**The recipe:** `tools/quantize_qwen38_tuned.sh`'s flags (`--scheme W4A16 --group_size 64 --quant_lm_head --format
auto_round:auto_gptq --low_gpu_mem_usage`), AutoRound 0.17.0 @ `6afaecdb` + `auto-round-local.patch` via
`tools/quantize/ar_pin.sh` (`tools/quantize/README.md`). Three things to prove on the tiny model before the 360 GB run
(plan 21v2a Task 1):
- **the `lm_head` packing.** On 0.17.0 the tuned path packs the head; the RTN path in low-CPU-memory mode does **not**
  (it ships `lm_head.weight` unpacked while `extra_config` says 4-bit) and its workaround,
  `--disable_low_cpu_mem_usage`, holds the whole model in RAM - impossible at 360 GB on 115 GB. So an RTN run packs the
  head by our own RTN g64 (the loader's rule, `loader/rtn.h`) into the export, checked by `check_gptq_export.py`;
- **per-expert export from the fused original** (`mlp.experts.gate_up_proj [512, 1280, 2560]`): per-expert
  `qweight` / `scales` / `qzeros`, what our loader reads (Intel's 0.15.0 export did it; prove the pinned commit does);
- **the MTP head:** transformers drops `mtp.*`, so AutoRound never sees it. Its experts go into the export either by
  a second, small AutoRound pass over the MTP layer (its inputs captured from the main model's last hidden on the
  calibration rows) or by RTN g64 (today's load-time rule, spec 15e) - Task 1 says which the toolchain can do, Task 2
  sizes both; acceptance against the bf16 head decides (§7.2).

**Sizing (an early task, not a guess):** 48 blocks of ~5.2 GB bf16 each (512 experts x 3 x 2560 x 640 x 2 B
= 5.03 GB + the dense part, derived), 24,576 routed experts; read from the cold disk at ~26 s a block (estimated,
~200 MB/s). The tuned run's cost per block is unknown on a B70; Kolibri's estimate was 10-30 h for 384 experts a
block on an H100. Plan 21v2a Task 2 measures the first block on a card (RTN, and tuned at 200 iterations) and on the
box CPU (which has no bf16 / AVX-512: fp32 only, `cpu-bf16-slow`), projects x 48 + the MTP head, and records peak RAM
and VRAM; the operator then picks tuned or RTN and the machine time (§12 question 2). Every expert must see
calibration tokens: top-10 of 512 routes ~2 % of tokens to an expert, so the per-expert coverage count of plan 21q
(floor 2048 routed tokens, proposed) is part of the recipe for a tuned run.

**Export size (derived):** experts 64.2 + MTP experts 1.3 + dense ~1.7 + embedding 1.3 + head 0.34 GB, plus the bf16
PLE table (102 GB) if AutoRound copies it through: ~70-172 GB, on the cold disk, moved hot by `tools/hf_tier.sh`.

**Quality of our export against Intel's** (plan 21v2a Task 7): KL and top-1 of each against the bf16 original on the
same held-out rows (golden prompts and the tool-call prompts; layer-streamed CPU reference, f32 arithmetic), and the
engine's own count rule and twin count against an f32 reference of our weights. Ours trades Intel's bf16 dense layers
for int4 g64 ones and gains g64 experts and a tuned head; if it comes out worse than Intel's on both KL and top-1, the
work stops for the operator (§12 question 3).

## 3. The bytes (derived)

**Geometry, checked against `src/model/qwen4exp.h` and `src/loader/qwen4exp_layout.h`** (and the same in basalt's
`decode/layout.hpp` and Strata's `core/geometry.hpp`): hidden 2560, 48 layers = 36 GDN + 12 QSA (every 4th, l % 4 ==
3), 4 hyper-connection streams (10240), 512 routed experts top-10 of width 640 plus a shared expert of width 640, QSA
24 q / 2 kv heads x 256, an indexer of 4 heads x 128, one PLE layer (layer 1) over a 320,001,536 x 160 table, one MTP
layer with its own 512 experts. A routed expert in our layout 1 is 2,611,200 B (gate‖up 1,740,800 + down 870,400; 4.25
bits a weight with its f16 g64 scales); 512 of them are 1,336,934,400 B a layer.

| per decode token, GB | ours (int4 g64) | Intel's |
|---|---:|---:|
| non-routed layer weights (36 GDN + 12 QSA: projections, shared expert, router, hyper-connections) | **3.04** (36 x 63.7 MB + 12 x 62.0 MB) | 7.31 (36 x 155.6 + 12 x 142.4) |
| of which hyper-connections (bf16 in both) | 1.29 | 1.29 |
| head and fixed (`lm_head`, PLE block 0.066, final mixer 0.013) | **0.42** (int4 head 0.338) | 0.72 (int8 head 0.637) |
| **D, non-routed** | **3.46** | 8.03 |
| **R, routed** (48 x 10 x 2.611 MB) | 1.253 | 1.253 |
| **total** | **4.71** | 9.28 |

- **Roofline** at 590 GB/s, every expert in VRAM: ours 125 t/s, at the 77.6 % of roofline Qwen3.8 reaches **97 t/s**
  (Intel's form: 64 / 49). This is spec 21 §3's figure (119 / 93 with the int8 head) come back.
- **Weights in VRAM for the whole model (ours):** experts 64.17 + non-routed 3.04 + embedding 1.27 + head and fixed
  0.42 + the MTP head 1.52 (its dense part bf16, its experts int4) = **~70.4 GB** (Intel's: ~75.0), against ~62 GB
  usable on two cards (32.53 GB each less the 1.5 GB reserve).
- **Resident fraction f** (two cards, 32k, the MTP head and the prefill scratch, the same 4.8 GB of KV and scratch the
  planner's `expert_budget` gave for Intel's form): about **13 GB of experts outside VRAM, f ≈ 0.80** (derived, rough;
  0.724 on Intel's form). The planner computes the exact value once it knows our forms.
- **Hit rate h (estimated before the curve):** routing is flat - basalt measured the hit share at about f^(1/6)
  (25 % resident -> ~79 % of reads, 50 % -> ~89 %), so f ≈ 0.80 gives h ≈ 0.96 static; saber's online policy reached
  93.7 % resident hits on held-out traffic where a static placement got 79.7 %. Plan 21v2a Task 4 measures it on our
  traces.
- **KV:** 25,344 B a position (12 QSA layers + indexer keys): 0.83 GB at 32k, 6.6 GB at 262,144 - each GB of KV is a GB
  of experts (spec 22 decision 5, still true).

## 4. The yardsticks, honestly, and why we can beat them

**Quantisation class decides the speed** - fewer bits are fewer bytes a token and more experts in VRAM. Strata's own
quality table (top-1 against Q8_0 / mean KL): UD-Q4_K_XL (4-bit class) 93.1 % / 0.056; IQ3_S 87.7 % / 0.144;
IQ3_XXS 84.5 % / 0.202; Q2_0 80.6 % / 0.315. Our int4 g64 AutoRound is the 4-bit class.

| engine | cards | weights | decode t/s | prefill t/s | (published) |
|---|---|---|---:|---:|---|
| Strata | one B70 (PCIe 4, 13.3 GB/s, 60 GB RAM) | IQ3_S, ~2.5-3 bpw mixed; 11.5k of 24.6k experts resident | 30 (prose) - 41 (code) | ~1000 at 4k | |
| Strata (community) | two B70, layer split | IQ3_XXS | **66** | 394 | the goal (§0.3) |
| Strata | two B65, all in VRAM | IQ3_S | 71-83 | 1261 / 1511 / 1628 at 4k / 8k / 16k | |
| XeStrata | one B70 | Q2_0 / IQ2 / IQ3 | 70-96 (1k-4k), 50-63 at 262k | | |
| XeStrata | one B70 | **UD-Q4_K_XL (4-bit class)** | **20.8** | | |
| XeStrata | four B70, layer split | **UD-Q4_K_XL** | **72-78** | 1111 / 1869 / 2233 at 4k / 32k / 127k | 93-94 % cache hits |
| saber | two AMD R9700 (bandwidth like 2x B70) | MXFP4 experts W4A8, fp8 trunk | 162-168; ~389-420 at 8 requests | ~7000 at an 8k chunk | |
| basalt | two RTX 5080 / one 5090 (NVIDIA) | NVFP4 | 358 / 395 | | |

So on B70s at 4-bit the references are **20.8 t/s on one card and 72-78 on four**; Strata's 30-41 and 66 are 2.5-3 bit
numbers. Our derived bound for two cards at 4-bit is 75 (`pp`) to 118 (`tp`) t/s at h = 0.97 (§5) - beating 66 at
better quality is plausible, not given.

**Why we can beat them:**
1. **P2P in the hot path.** Nobody else uses GPU-to-GPU traffic per token (basalt stages through the host; Strata and
   XeStrata split by layers). Our tensor parallel all-reduce over P2P costs 3.5 µs inside replayed lists (measured on
   Qwen3.8), which makes `tp` with every expert split (§5 (b)) cheap: both cards' bandwidth on every layer, both PCIe
   links streaming misses at once.
2. **Captured lists, fewer launches.** Strata runs ~2,400 graph nodes a window and reaches 35-40 % of its bandwidth
   bound; our decode is one replayed list (779 launches a token on this model, spec 21 §14) and Qwen3.8 reaches
   77.6 % of roofline. Routing is already device-planned: the expert kernels read ids and (with the tier) addresses
   from device tables, no host action inside a step.
3. **An all-pinned host tier with ~115 GB.** Every byte a step reads from the host is pinned and read in place - the
   int8 PLE table (51.8 GB) and the misses mirror (~13 GB, §6) fit beside each other. Strata's box has 60 GB and must
   fall back to disk reads for the PLE; we need no disk path and no mmap on the token path.
4. **Better bytes where they count:** 4.25-bit experts with tuned g64 scales at ~5 % fewer bytes than a 4.5-bit K-quant,
   and an int4 head (0.34 GB against 0.64 int8).

What we cannot change: PCIe 3.0 (12 GB/s a link, measured host-to-device on device 0, against Strata's 13.3 on PCIe 4).

## 5. The second card: layer split first, then tensor parallel with the expert split

The cost model is spec 22 §2's with the VRAM part at Qwen3.8's 77.6 % of roofline (457.8 GB/s), P = 12 GB/s per PCIe
3.0 x16 link, D and R from §3 (ours), h the share of routed bytes found in VRAM:

- **(a) layer split + tier (`pp`, built; first by §0.2).** Each card caches its own layers' experts; misses stream from
  pinned host memory over that card's link; the cards decode one after the other, so one link works at a time.
  `t = (D + h·R) / 457.8 + (1 - h)·R / P`.
- **(b) tensor parallel with the expert split + tier (`tp`, plan 21v2b).** saber's served default (tp2): each card
  holds **half of every expert along the 640 width** (320 a card: gate‖up split by columns, down by its K - 5 groups of
  64) and half of every dense projection (QSA: 12 q heads and 1 kv head a card; GDN: 8 k / 24 v heads a card);
  hyper-connections, routers and the indexer replicated (1.29 GB a token on each card); both cards compute all ten
  routed experts and the shared one, and their partial sums ride the post-MoE all-reduce TP already does - no extra
  traffic. Two all-reduces a layer over P2P (~98 a token x 3.5 µs = 0.34 ms). saber measured this against expert
  parallel (whole experts per card): one request 162 -> 167 t/s, 8 requests 414 -> 420, MoE imbalance 2.42 -> 0.06 ms.
  Both links stream misses at once, each card only its own halves. D per card ≈ 1.29 + (3.46 - 1.29) / 2 = 2.38 GB,
  R per card 0.627 GB. `t = (2.38 + h·0.627) / 457.8 + (1 - h)·0.627 / P + 0.34 ms`. f per card about as in (a).
- **(c) card 1 as a peer expert tier** (basalt's design: card 0 runs everything, cold experts live in card 1's VRAM and
  are read over P2P, or card 1 computes the experts it holds). basalt measured this beating a layer split on NVIDIA
  with a small dense part; with our int4 dense part (3.46 GB) it is closer than it was with Intel's 8 GB, but every
  token still sends most routed bytes or activations across the bus, and (b) uses both cards' bandwidth on every
  layer. Not built unless the probe's P2P read rate (plan 21v2a Task 5) is far above the host link's.

| h | (a) `pp` + tier t/s | (b) `tp` + tier t/s |
|---:|---:|---:|
| 1.00 | 97 | 145 |
| 0.97 | 75 | 118 |
| 0.95 | 65 | 106 |
| 0.90 | 49 | 83 |
| 0.85 | 39 | 69 |

(derived, ours; MTP off, one request, short context; no swap gains, P is the copy rate - an in-kernel read of host
memory may be slower, measured by the probe; the 77.6 % already holds Qwen3.8's launch overhead, Flash-Next's own is
not measured.) XeStrata's pipeline windows (+17-28 %) apply only to a layer split; they are a lever for (a) if (b) is
late.

## 6. The tier

Spec 22 §4's design holds - a byte-sized VRAM cache of whole experts (under (b): whole expert halves) per layer, filled
from a routing profile; a device-side table `[layer][1024]` u64 (512 gate‖up, 512 down pointers) read by the expert
kernels through the existing macro pair `Q4_EXPERT_GU` / `Q4_EXPERT_DN` (`q4_moe.cl`, `q4_pf_moe.cl`); no host action
between layers; the captured lists do not change per step.

**Misses are read in place from pinned host RAM** (saber's "L2", Strata's host mirror; spec 21's PLE gather
mechanism): the GPU kernel reads the host copy over PCIe, no copy step. saber reads a 2.6 MB record in ~90-96 µs.
Strata needed exactly this on Xe because a host / GPU hand-off inside a step is unreliable.

**Pinned pieces and the pin question.** The mirror is per layer (a layer's non-resident experts: ~0.27 GB at
f ≈ 0.80, at most 1.34 GB), in pieces of at most 2 GiB; under (b) each card maps only its own halves. Any range mapped
by both cards stays at or below 4 GiB. XeStrata saw a Level Zero pin limit of ~30 GiB per process (its
`STRATA_PIN_PARTS`); our loader already pins the 51.8 GB PLE table in 16 ranges on one card (`r31.load` passes), so
the open part is the **total with both cards mapping their own pieces** (~65 GB): plan 21v2a Task 5 pins it piece by
piece up to the target and reports the first refusal or hang.

**Host RAM (~115 GB usable, §0.5):** PLE int8 51.8 + misses mirror ~13 = ~65 GB pinned for the whole-model GPU job;
the prefix cache takes what is left by its auto rule (~30-40 GB, estimated); a CPU job beside it only if it fits in
the rest. A full mirror (64.2 GB + 51.8 = 116 GB) does not fit and is not needed.

**Residency policy:**
- **Warm start from a recorded profile** (saber `route_profile.json`, Strata / XeStrata shipped profiles, basalt's
  `48x512.prof`): ours is `<snapshot>-expert-profile.safetensors` from `offload_curve.py profile`, beside the
  checkpoint. No flag: the engine fills each card's cache from the profile and the planner's budget; without the file
  it fills by id order and says so at startup.
- **Online swaps are likely to pay** - saber's static placement lost on held-out traffic (79.7 % vs 93.7 % resident
  hits) - so the trace simulation (plan 21v2a Task 4) tests these policies, and the best one by h at equal link bytes
  is built: basalt's (counts decayed by 0.85 a round, adapt every window, up to 96 swaps, a candidate needs the
  victim's count + 1.5, peer margin 10 routings, 4x burst when saturated, seeded from the prompt) and saber's (at most
  2 inserts a layer a step; **second-chance insert** - promote on the second host miss within 8 steps: +5 % at 8
  requests and -20 % link bytes, bit-exact; a layer routing more than 50 % of its slots reads through; next-step
  prefetch off, it hit 4 %).
- **Copy scheduling:** promotions copy in 1 MiB pieces on the copy engine in idle bus time between steps, and the
  table entry moves only after the copy's event (spec 22 §4's four steps), so the captured list never waits. basalt /
  saber go further - copies issued after the next layer's head launch and a per-layer "landed word" the next window's
  kernel polls - which needs device spins on host-written flags (Xe hazards, §8); taken only if between-step copies
  cannot keep up, with system-scope atomics and bounded spins. Tests pin the placement (an environment variable) so
  residency gates stay bitwise.

**The PLE from host RAM, never in VRAM** (as every engine does). Ours reads the 16 rows a token in place inside layer
1's kernel. The others prefetch the rows as soon as the token (or the draft token) is known, gather them on a host
thread and upload 10 KiB for layer 1, hidden behind the embedding and layer 0. `r32.ple_rate` measures ours first; if
the in-kernel gather shows in the step time, the gather moves before the step's list is submitted (between steps,
allowed), into a small pinned buffer.

**Prefill, two modes** (Strata / XeStrata): below a crossover (theirs ~2.5-6k rows) only the routed experts' misses
are staged; above it every non-resident expert streams through a staging ring (theirs 96-384 expert slots) on the
copy engine, overlapped with the previous layer's attention - a GEMM must never read host memory once per tile. Bus
cap (derived, ours): a 2048-row chunk touches nearly every expert, ~13 GB a chunk at f ≈ 0.80, 1.1 s at 12 GB/s -
~1900 t/s on one link, ~3800 with both links streaming (`tp`). A look-ahead router copy of the next layer's likely
experts is a later refinement.

**An expert pack beside the checkpoint** (saber's xtier: a fixed per-record table and 4096-aligned data) is not needed
for our g64 export (the loader expands Intel's g128 only); built only if load time is a problem.

**Not taken:** CPU-computed misses (XeStrata, basalt: our replayed lists have no host hand-off inside a step), pruning
(removed), expert parallel by id halves (saber measured the expert split better, (b)).

## 7. Decode levers after the whole model runs

1. **The launch count is the lever** (Strata: ~2,400 nodes a window -> 35-40 % of the bound; basalt: one graph per
   window size, device-planned routing, in-kernel spins on mapped flag words bounded at 200 ms, a persistent kernel on
   the peer card). Ours: 779 launches a token; measure Flash-Next's fraction of roofline first, then fuse where it
   shows: the three launches a gated residual, the five a QSA layer (prep, score, select, attention, reduce).
   Shared expert: already folded in (slot 10 of the 11-slot `q4_moe_gate_up` / `q4_moe_down`, block 512 of the grouped
   prefill GEMM); Strata found a separate stream fork costs more than it saves on Arc.
2. **MTP policy.** Published acceptance on this model: 66-71 % on prose and conversation, 90-99 % on code and
   structured output (Strata, basalt); verify windows up to 8; basalt keeps draft j while the product of the
   per-position acceptance estimates is ≥ 0.35, with a cost-per-window EMA. Open-work item 13 builds the same per-step
   choice for Qwen3.8 - reuse it. The MTP experts are int4 g64 in our export (§0.4); `r34.accept` measures acceptance
   with our head against the bf16 head (Strata keeps its MTP experts at Q2_0, 708 MB, all in VRAM; saber RTN-quantises
   them to MXFP4). The head stays resident.
3. **Union-aware verify:** the MTP verify at M = K + 1 rows reads each row's experts; reading each distinct expert once
   a verify (Bolt's union MoE, spec 26) cuts the routed bytes, and under the tier the misses too.
4. **QSA, with OpenVINO as the Intel reference.** OpenVINO `3b0ee89623` added fused GPU ops for QSA (the indexer and
   the sparse attention, paged) with a quantised KV layout: the reference for a fused decode kernel and for int8 KV on
   the 12 QSA layers (int8 KV frees ~3.3 GB at 262k for experts); not its paging (fixed slots, ruled).
5. **int8 hyper-connections (operator 2026-11-12: do it):** the 4-stream mixers stay bf16 in the export and are converted
   to int8 **at load** (one fp16 scale per 32 values, basalt's and saber's "hc8"), as the output head is today. 1.29 ->
   ~0.65 GB a token: 4.71 -> ~4.06 GB, the decode ceiling ~97 -> ~112 t/s (derived). basalt reports ~0.006 KL for it.
   Gate: the f32 judge on our export (twin count no worse than with bf16 mixers) and the measured decode gain.
   **If accuracy is bad, suspect these first:** the mixers touch every token's residual stream, so they are the first thing
   to switch back to bf16 when a quality gate fails - a load-time choice, no re-export needed - before suspecting the int4
   weights, the int4 head or the MTP experts.
   **Test switch (operator 2026-11-12):** an environment variable, `B70_Q4_HC=int8|bf16` (default `int8` once the lever
   passes its gate), picks the mixers' form at load - for A/B timing and quality checks only, not a user flag; listed in
   docs/environment-variables.md when it is built.
6. **Prefill GEMM - built on Ornith first (operator 2026-11-12):** Flash-Next's prefill reuses Ornith's grouped expert GEMM
   (`pf_moe_gemm.cl`) and the same separate unpack pass (`q4_pf_dequant_*`), so the Ornith MoE prefill rework - int4
   unpacked in registers inside the grouped GEMM, bigger MoE chunks and tiles, persistent scheduling - lands here unchanged.
   Also: int8 DPAS (s8) grouped GEMM decoding the int4 weights in registers (Strata ESIMD, +12-21 % below
   ~130 rows an expert; our l0-int8 prefill is the base), XeStrata's strip-ordered work-group numbering for wide GEMMs
   (105 -> 145 TFLOP/s on Xe2).

## 8. Xe hazards (Strata, XeStrata, ours)

- VRAM over-allocation spills to host RAM and can livelock the box - the planner never over-commits.
- Aliased 2 MiB pages in large device arenas (Strata's `malloc_device_guarded`) - the probe's alias check
  (`r36.alias`) covers our arenas.
- Unbounded device spins reset the GT - every device-side wait is bounded.
- Host stores during a running kernel are not reliably visible - system-scope atomics for any flag a kernel reads.
- A copy or graph submitted while the host waits inside a window stalled (XeStrata) - copies go between steps.
- Ours: a host allocation above 4 GiB mapped by both cards hangs the driver (2026-11-11) - 2 GiB pieces.

## 9. Context against cache

Zero-config: the planner gives the cache every byte left after the non-routed weights, the scratch and the KV of the
chosen context. `--max-len auto` for the whole model is the largest of 32768 / 65536 / 131072 / 262144 whose projected
decode stays within 5 % of 32768's (spec 22 decision 5's rule, proposed); `--max-num-seqs N` slots follow the
`--max-num-seqs` design (context auto-fits N slots).

## 10. Quality bar (the f32 judge; approved 2026-11-12)

- **The tier changes no arithmetic:** with every expert resident it is bitwise today's engine (O0); forced residency
  of 0 / 50 / 100 %, random and adversarial, is bitwise the 100 % run in decode, prefill and verify (O1); swaps every
  step are bitwise the no-swap run (O2). A difference is a bug, never a tolerance.
- **Our export against Intel's and the bf16 original** (§2): KL and top-1 per held-out set, both checkpoints on the
  same rows.
- **The whole model against an f32 reference of our weights:** a CPU f32 reference of all 48 layers on `q4exp_4k`
  (37 layers peaked at 68 GiB, so ~90 GiB estimated: it holds both cards' lock as a > 64 GB job). Prefill by the
  operator's count rule (its rejected routes ≤ 1.2x decode's, its f32-accepted tokens ≥ decode's).
- **Decode's own bar, the twin count:** the engine's f32-rejected tokens and routes on the golden prompts at most the
  bf16 CPU reference's own on the same prompts (as for tensor parallel). Under `tp`, the same rule against `pp`.
- **The tool-call test:** the 36 scenarios on the whole model against the CPU reference (`r34.a4_ref`), every first
  difference judged by the f32 judge (`serve_judge.py`), as for Qwen3.8.


**When a quality gate fails, check in this order:** (1) the int8 hyper-connection mixers (§7 lever 5) - switch them back
to bf16 at load and re-judge; (2) the int4 output head (against the f32 reference's head); (3) the MTP experts (acceptance, `r34.accept`); (4) the int4 g64
dense layers themselves.
## 11. Speed gates (approved 2026-11-12) and the goal

| stage | decode, one request, 4k depth, MTP off | prefill at 4k |
|---|---|---|
| (a) `pp` + tier, the whole model served (plan 21v2a) | **≥ 30 t/s** | recorded |
| (b) `tp` + tier (plan 21v2b) | **≥ 50 t/s** | **≥ 1000 t/s** |
| **the goal** | **> 66 t/s** (Strata, two B70s, IQ3_XXS) at int4 g64 | |

**Stopping rule:** if the probe's projection at the measured h gives (a) < 30 t/s and (b) < 50 t/s, stop before
engine work and bring the levers to the operator (§12 question 1). If the gates pass and the goal is missed, §7's
levers go in order of measured gain.

## 12. Open questions for the operator

1. **Base format (recommended, not decided):** int4 g64 for every expert, as the one format; mixed 3/4-bit experts
   (Strata's way to fit two cards: 3-bit for the cold experts would put nearly all of them in VRAM) only as a later
   lever if the gates are missed, judged by the f32 judge. Recommendation: yes - int4 g64 now, the mixed form only on
   evidence.
2. **The quantisation run's machine time,** after plan 21v2a Task 2's sizing: tuned AutoRound or RTN, and how long a
   card (or both) may be held for it. Recommendation: tuned if the projection fits in about three days on one card
   (the other card keeps working), else RTN with the head packed by our rule.
3. **If our export loses to Intel's** on both KL and top-1 against the bf16 original (§2): keep ours anyway (one
   format), or keep Intel's bf16 dense layers beside our g64 experts? Recommendation: decide on the numbers; the f32
   judge and the tool-call test are the tie-breakers.
