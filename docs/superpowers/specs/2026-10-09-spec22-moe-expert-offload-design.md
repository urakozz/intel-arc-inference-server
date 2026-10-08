# Spec 22 - the MoE expert-offload tier

**Status:** outline, probe-first, 2026-10-09; the operator approved the design on 2026-10-08. This document gives
P0 in full, the design as a sketch and the decisions P0 feeds, marked **(decide)**. The design is detailed after
P0's record. Nothing is built.

**Why:** Qwen3.8-Flash-Next (spec 21) does not fit two B70s at int4 g64:
- routed experts 64.17 GB + ~5 GB of everything else + the MTP head + KV, against ~62.2 GB usable (spec 21 §3);
- **its full model runs end to end only once this spec lands.**

The tier keeps a byte-sized cache of experts in VRAM and lets the GPU read the misses from a pinned host mirror
over PCIe:
- **No CPU expert compute**, and no host doorbell per layer: everything stays inside the captured lists.
- The mechanism is a community MIT engine's Arc design (Strata, `docs/INTEL.md`, "Arc Pro B70 with a model that
  does not fit", 2026-10-07), rebuilt inside our replayed-list model.

**Order:** after spec 21c (decode on the truncated model) for the engine work. **P0 can run as soon as spec 21a's
traces and a box day exist**, and it uses spec 21's PLE gather as its first zero-copy client.

Every number is **measured** unless marked **derived**, **estimated** or **published**.

## 1. The facts it starts from

- **The box is a Dell T5810** (one CPU socket, **PCIe 3.0 only**), whatever its hostname says. The two B70s may
  train at x8. Nothing in the repo records their link widths.
- **Measured pinned copies on device 0** (`docs/probe-prefix-cache-2026-09-27.md` §2): host to device 12.2-12.9
  GB/s, device to host 14.2 GB/s, on an immediate list, 128 MiB to 3.9 GB. That is an x16 Gen3 rate; device 1 was
  never measured, and neither was a GPU kernel reading host USM directly.
- **Measured pinned allocations:** 48 GiB worked, the largest tried; the cap was MemAvailable - 16 GiB = 101.2 GiB
  (same doc §1). **Box RAM is about 128 GB** (docs/10). The full expert mirror (64.17 GB) plus spec 21's int8 PLE
  table (~51.8 GB) is ~116 GB pinned, which is above that cap (decision 4).
- **Expert sizes (derived):** one expert at int4 g64 is 2.611 MB (gate‖up 1.741 MB, down 0.870 MB). A layer's 512
  experts are 1.337 GB. A token's routed bytes are R = 1.253 GB (48 x 10 experts).
- **Community measurements on the same card (published):**
  - Strata's B70 run had 11.5k of 24.6k experts resident with the rest in a pinned mirror read over PCIe, at 13.3
    GB/s host to device measured in a VM. Decode was 30-41 t/s with MTP on a different quantisation.
  - A 4K prompt touched nearly all 512 experts of every layer: about 26 GB over the bus per chunk.
  - **The xe driver mapped two 2 MiB pages of a 22 GiB device arena onto the same memory.** The engine now tags and
    reads back every allocation of 32 MiB or more.
  - **A per-layer host / GPU handshake was not visible across the bus on xe** (NaN logits without its "no host"
    mode).
  - Its paper (§3.4-3.5, Fig. 3) reports a profile-filled cache's hit rate against cache size, 0.50 at its 12 GB
    card rising to 0.72 measured with adaptive swaps.

  The last three points argue for this design (a device-side table, no per-layer host handshake) and for P0's alias
  check. None of these numbers is a bar for ours.

## 2. The cost model (derived)

Per decode token, with D = 3.712 GB of non-routed bytes (spec 21 §3's 4.97 GB without R), BW the VRAM rate, P the
PCIe read rate, and h the hit rate (the share of routed bytes served from VRAM):

- **serial:** `t = (D + h·R) / BW + (1 - h)·R / P`
- **overlapped (lower bound):** `t = D / BW + max(h·R / BW, (1 - h)·R / P)`, if a layer's misses stream while its
  hits compute

At BW = 590 GB/s (the roofline) and P = 12 GB/s (R / P = 104.4 ms):

| h | serial ms | serial t/s | overlapped t/s |
|---:|---:|---:|---:|
| 1.00 | 8.42 | 118.8 | 118.8 |
| 0.99 | 9.44 | 106.0 | 119.1 |
| 0.95 | 13.53 | 73.9 | 86.9 |
| 0.90 | 18.65 | 53.6 | 59.8 |
| 0.80 | 28.87 | 34.6 | 36.8 |
| 0.50 | 59.56 | 16.8 | 17.1 |

- **The hit rate is the whole game:** every point of miss rate costs ~1 ms a token. P at x8 (~6 GB/s) doubles the
  miss term.
- **Under spec 16's pipeline the two cards decode in sequence**, so one link is busy at a time; a prefill chunk
  pipeline loads both.
- **Prefill (derived):** a 2048-row chunk touches ~all experts, so it streams `(1 - f) x 64.17 GB` at resident
  fraction f. At f = 0.8 that is 12.8 GB, ~1.07 s at 12 GB/s: a bus cap near 1.9k t/s at chunk 2048 and 3.8k at
  4096, before compute.
- **The resident fraction available (derived, two cards):** ~62.2 GB less ~5 GB of everything else, the MTP head
  (~1.4 GB), scratch and KV. That gives f ≈ 0.83 at 32k context, 0.79 at 128k and 0.74 at 262144. **h ≥ f** with a
  profile fill; the routing skew decides how much more.

## 3. P0 - the probe (in full)

Idle box, the GPU lock for both cards, interleaved pairs after warm-up (the repo's protocol), medians of 5. The
probe is a standalone tool (`tools/probe/probe_offload.cc`) plus a trace analyser (`tools/oracle/offload_curve.py`);
the record goes to `docs/probe-offload-<date>.md`.

**P0.1 The links.**
- `lspci -vv` `LnkCap` / `LnkSta` for both B70s (04:00.0, 08:00.0), the bridges above them (the cards' on-board
  switch upstream ports) and their root ports. Record the generation and width each trains at, idle and under load.
- The CPU model, the DIMM population and the memory channels (`dmidecode`), from which the host RAM bandwidth
  ceiling follows.
- **Stop and report** if either card trains below Gen3 x8.

**P0.2 Copy-engine host to device.**
- Pinned host USM to device memory, on an immediate list and inside a replayed list.
- Sizes: one expert's down (0.87 MB), gate‖up (1.74 MB), a whole expert (2.61 MB), a layer's typical miss set
  (16 / 64 / 256 experts) and 1 GB.
- Per card, then **both cards concurrently** (a prefill pipeline's case).
- Also the first-byte latency of a single 2.6 MB copy.

**P0.3 GPU zero-copy reads from host USM** - the design's miss path:
- A kernel reading a host USM range with **the expert GEMV's own access pattern** (int4 layout-1 blocks, the
  `q4_moe_gate_up` work-group geometry, M = 1).
- Sizes as P0.2. Per card, then both concurrently.
- **Mixed launches:** a launch where 10 of 11 slots read VRAM and 1 to 3 read host, against all-VRAM. Does the launch
  take max(VRAM part, host part) or their sum? That decides between §2's two formulas.
- The same inside a replayed list, repeated 10^4 times, checking the bytes every time (the xe visibility hazard).
- **The comparison arm:** copy-engine staging of the misses into a VRAM bounce buffer, then the all-VRAM kernel.
  This is decision 1's evidence.

**P0.4 Host RAM under load.**
- The host memory bandwidth with both cards zero-copy reading and the CPU idle.
- The same with the CPU running a decode-sized host task (the server's own work).
- This is the ceiling P0.3's concurrent number must stay under.

**P0.5 Pinned capacity and integrity.**
- The largest pinned host allocations as one range, and as 48 per-layer ranges plus the PLE table's shards.
- Targets: 64.17 GB of experts plus 51.8 GB of int8 PLE; record where it fails and what MemAvailable is.
- A tag-and-readback alias check of a ≥ 22 GiB device arena allocated after a model-sized allocation history, and
  of the pinned host ranges as the GPU sees them.

**P0.6 The hit-rate curve from routing traces** (the analyser, on spec 21a's traces).
- **Source:** teacher-forced routes, one layer-major forward over each recorded transcript on the box CPU reference
  (Intel's checkpoint): the A4 tool-call scenarios, the opencode recording, code, prose. The engine with `--layers
  N` gives exact routes for layers [0, N) on real weights (a layer depends only on the ones before it), a
  cross-check for those layers.
- **Static fill:**
  - per-layer per-expert counts;
  - a profile built on one half of the transcripts and scored on the other (held out);
  - the curve of h against the resident fraction f, with the cache sized in **bytes per layer**, uniform against
    from each layer's own curve.
- **Adaptive swaps,** simulated on the traces: swap up to B bytes every S steps; an incoming expert is used only
  from the step after its copy lands (P0.2's measured copy time); its slot is reused only after the evicted
  expert's last in-flight step.
- **MTP verify:** the union of experts over M = K + 1 rows per layer (K = 1..4), the miss count per verify step.
- **Prefill:** unique experts per layer per chunk (2048, 4096); the streamed bytes at f = 0.6..0.9.

**P0.7 The projection.**
- From P0.1-P0.6: decode t/s at 4k / 32k / 128k (f from the context's KV, h from P0.6, the formula P0.3 supports),
  with and without MTP (acceptance from spec 21e when it exists, else a range).
- Prefill t/s at chunk 2048 / 4096.
- One card (f ≈ 0.34) as a row for completeness.

**P0.8 The pruning curve** (for decision 9, the pruned mode; the operator's idea, 2026-10-09). From spec 21a's
CPU reference on the agentic set: score every (layer, expert) by its routed contribution - REAP's criterion, the
router weight times the norm of the expert's output, summed over the trace's tokens (frequency alone is a weaker
proxy) - then, for kept fractions 50 / 62.5 / 75 / 87.5 % per layer, re-run the reference with the dropped experts
masked out of the router (logits -inf before the top-10, the top-10 renormalised over the kept set) and record the
KL to the full model and the argmax agreement on the golden and A4 sets, plus A4 tool-call accuracy at the kept
fractions that fit the cards entirely (50 % = 32.1 GB at int4 g64). A published reference point: a community
coding-only GGUF keeps half the experts (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`, reported weaker
outside code); its kept set is not published as original expert ids, so ours is computed, not copied.

**Stopping rule:** if the projected decode at 32k is below 40 t/s on two cards with the best fill, stop and report
before any engine work. The tier's levers then come first in the record (HC int8 to free D, int8 KV, a smaller
context).

## 4. Design sketch

- **The cache.**
  - A VRAM arena per card holding whole experts in slots sized in **bytes per layer**, not a uniform slot count:
    a layer's budget follows the P0.6 curve.
  - Filled at load from a **routing profile** (per model, shipped beside the checkpoint; the default when none
    exists is the profile from P0.6's traces).
  - The arena passes the alias check at load.
- **The host mirror.** Every expert in pinned host USM (per-layer ranges), checksummed at load. It is the source for
  misses and for swaps.
- **The indirection table** (device memory, `[layer][512]`):
  - each entry is a VRAM slot pointer or the host USM pointer of that expert;
  - spec 21's `q4_moe_gate_up` / `q4_moe_down` read their 11 slots' weights through it instead of `id x stride`;
  - **the captured list does not change, and no host action happens between layers.**
  - An all-resident model (spec 21's truncated runs) is the special case of every entry pointing to VRAM.
- **The miss path:** the expert kernels read a missed expert straight from host USM (zero-copy) - spec 21's PLE
  gather mechanism, reused - unless P0.3 shows copy-engine staging wins (decision 1).
- **Adaptive swaps, between steps only:**
  1. the host decides from device-written per-expert hit counters (read once every S steps);
  2. it points the victim's entry at the host first;
  3. after the step that last used the slot has retired, it copies the incoming expert into the slot on the copy
     engine;
  4. **it admits the expert (entry → slot) only once that copy's event has signalled.**

  A step never waits for a swap; a swap never races a reader.
- **Prefill:** for each layer, the chunk's uncached experts are copied into staging (borrowed cache slots or a
  dedicated ring, decision 7) on the copy engine **while the previous layer computes**; the grouped GEMM reads
  through the same table. Under spec 16c's chunk pipeline both cards stream at once (P0.2's concurrent number).
- **The embedding** (1.27 GB) can move to host USM like the PLE table (one row a token, zero-copy), returning its
  VRAM to the cache; an option, measured by the same P0.3 numbers.

### 4b. The pruned mode (lossy, opt-in, coding only) - the alternative and the complement

Instead of fetching cold experts, drop them: a per-model mask of kept (layer, expert) pairs, chosen by P0.8's
scores on our agentic traces. The router's logits for dropped experts are set to -inf before the top-10 and the
kept weights renormalised (the same softmax-top-k-renorm rule as today, over the kept set), and the dropped experts
are never loaded. At 50 % kept (256 per layer) the experts are 32.1 GB at int4 g64 and the whole model fits two
B70s with room for KV - no host tier, no PCIe on the decode path, about the all-resident speed (spec 21 §D).

- **It is lossy**, unlike the cache: output changes wherever a dropped expert would have been routed. **The
  operator's ruling (2026-10-09): only as a coding mode.** The mask is calibrated on coding / agentic traces only
  (the A4 set and opencode recordings), shipped as a named coding mask beside the checkpoint with its provenance
  (traces, criterion, kept fraction), selected explicitly (`--expert-mask coding`), and never the default: the
  general-purpose configuration is always the full model through the cache. Every gate records the KL against the
  full model (P0.8's curve is the bar), and the coding mode's own bar is A4 tool-call accuracy on code.
- **It composes with the cache:** a mild mask (e.g. 87.5 % kept) cuts the host mirror and the miss bytes without
  the full 50 % loss; the device-side indirection table simply has no entry for a dropped expert and the router
  never selects it.
- **Engine cost:** small - a mask applied in the route kernel (one bit per expert per layer) and a loader that
  skips dropped experts; no new kernel family.
- **It does not replace re-training:** experts are dropped, not merged or fine-tuned (merging methods are out of
  scope).

## 5. Gates (sketch)

- **O0, nothing moves:** with every expert resident, the tier is bitwise spec 21's engine (ids, logits, state).
- **O1, residency does not change arithmetic:** the same bytes read from host or VRAM give **bitwise** the same
  results. On a truncated real model that fits, forced residency of 100 / 50 / 0 % (random and adversarial sets)
  give bitwise equal ids and logits, decode and prefill.
- **O2, swaps:** a swap every step (a stress mode) is bitwise the no-swap run; a swap whose copy is withheld is
  never read.
- **O3, integrity:** the mirror checksums, the arena alias check, a hang-free refusal when a pinned allocation
  fails.
- **O4, speed:** decode and prefill on the full model against P0.7's projection (recorded; bars set from P0).

## 6. Stages (sketch)

- **22a. P0** (box, §3), its record and the projection; the operator's go / no-go.
- **22b. The table and the miss path:** the indirection table, the mirror, the static profile fill, O0-O1 on the
  truncated model. Then **the full model end to end on two cards**.
- **22c. Adaptive swaps and prefill streaming:** O2, the prefill overlap.
- **22d. Serving defaults and the record:** `--max-len auto` with the cache budget, the speed rows, O4.

## 7. Decisions P0 leaves open

1. **(decide) The miss path:** zero-copy kernel reads from host USM (proposed, the operator's design) or copy-engine
   staging into a VRAM bounce buffer. Evidence: P0.3, both arms, mixed launches.
2. **(decide) The cache budget per layer:** uniform bytes, or each layer's from its own curve (P0.6).
3. **(decide) The fill policy:** a static profile alone, or plus adaptive swaps (cadence S, bytes per round B) if
   P0.6's simulation shows they pay.
4. **(decide) The host RAM budget:** the expert mirror (64.17 GB) and the int8 PLE table (~51.8 GB) together exceed
   the measured pinned cap on ~128 GB. The options:
   - more RAM in the T5810;
   - mirror only the non-resident experts and exchange on a swap (a slot's evicted expert written back);
   - the embedding and PLE in one shared pinned budget.
5. **(decide) Context against cache:** each GB of KV is a GB of experts (f from 0.83 at 32k to 0.74 at 262144); the
   default `--max-len` follows P0.7's table (spec 21 decision 9).
6. **(decide) The MTP head's experts:** always resident (1.34 GB at int4) or cached with the rest.
7. **(decide) Prefill streaming:** borrowed cache slots (refilled after the prompt) or a dedicated staging ring;
   chunk 2048 or 4096.
8. **(decide) The profile's source and storage:** CPU-reference traces, engine traces, or both; per model, beside
   the checkpoint.
9. **The pruned mode (§4b): ruled a coding-only mode** (operator, 2026-10-09) - the default Flash-Next
   configuration is always the full model through the cache. **(decide)** the coding mask's kept fraction, from
   P0.8's A4 tool-call accuracy against P0.7's cache speed.

## 8. Out of scope

- CPU expert compute (the operator's design excludes it).
- An SSD tier.
- Changing the experts' format to make them fit (the pruned mode, §4b, drops experts; it does not change the format).
- Batching (spec 13) interactions.
- Models other than the `qwen4_exp` family until the tier exists; the indirection table is family-agnostic, so
  others can adopt it later.
