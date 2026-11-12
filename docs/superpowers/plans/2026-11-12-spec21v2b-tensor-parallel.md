# Spec 21 v2b - Qwen3.8-Flash-Next on `--parallel tp`: every expert split across the two cards, both links streaming

**Status (2026-11-12): planned.** Begins after plan [21v2a](2026-11-11-spec21v2a-whole-model-tiering.md) Task 10 (the
whole model served on `pp`), on our export. Short by design: the TP engine (specs 17b-17d, `src/runtime/tp_*`) and the
tier (21v2a) exist; this plan joins them.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** decode ≥ 50 t/s and prefill ≥ 1000 t/s at 4k on two B70s with tensor parallel, then past Strata's 66 t/s
(two B70s, IQ3_XXS) at int4 g64 - spec 21 v2 §5 (b), §11.

**Design (spec 21 v2 §5 (b), saber's tp2):** each card holds half of every routed and shared expert along the 640
width (gate‖up by columns, 320 a card; down by K, 5 groups of 64), half of every dense projection (QSA 12 q heads and
1 kv head a card; GDN 8 k / 24 v heads a card; `o_proj` / `out_proj` by K), and replicates the hyper-connections, the
routers, the indexer and the PLE projections; `lm_head` split by vocabulary as Qwen3.8's TP does. Both cards compute
all ten routed experts on their halves; the partial sums ride the post-MoE all-reduce (two all-reduces a layer over
P2P). Each card's tier caches and mirrors only its own halves and streams its misses over its own link. The PLE rows
are gathered once (card 0) and the 10 KiB result crosses over P2P - the table is never imported by both cards.

**Spec:** [spec 21 v2](../specs/2026-11-11-spec21v2-flash-next-v2-design.md). Precedents: plans
[17b](2026-10-05-spec17b-tp-decode.md) / [17c](2026-10-05-spec17c-tp-prefill.md) /
[17d](2026-10-05-spec17d-tp-integration.md) (Qwen3.8's TP: the split, the all-reduce in replayed lists, the twin
count).

## Global Constraints

Plan 21v2a's, unchanged (own worktree, branch `spec21v2b-tp` from main; signed commits, no trailers, never push; box
rules; GPU lock for every card job; ≤ 4 GiB per host allocation both cards map; F0; no host action between layers).
Zero-config: `--parallel auto` picks `tp` for Flash-Next only when it is measured faster than `pp`.

---

### Task 1: the split (host only)

- [ ] **Step 1:** `Q4Placement` gains the TP form; the loader's repack writes each card's halves (experts: gate‖up
  columns [0, 320) / [320, 640), down K groups 0-4 / 5-9; dense by heads / by K as above; the replicated parts whole).
  Host tests on the synthetic real-width checkpoint: the two halves concatenate bitwise to the one-card block; per-card
  bytes equal spec 21 v2 §5 (b)'s derivation (D 2.38 GB, R 0.627 GB a token).
- **Gate:** the host tests PASS; per-card weight bytes printed and within 1 % of the derivation.

### Task 2: decode on the halves (box)

- [ ] **Step 1:** the existing kernels at the half widths (`q4_moe_*` at an intermediate of 320, the dense GEMVs at
  half N or half K); new binaries by `tools/kernel_cmdlines` additions only.
- [ ] **Step 2:** the two all-reduces a layer through the TP engine's P2P all-reduce inside the replayed list; the PLE
  hand-off over P2P.
- [ ] **Step 3:** `--layers 18` (fits both cards with every expert resident), then the whole model with the tier.
- **Gate:** the twin count - `tp`'s f32-rejected tokens and routes on the golden prompts at most `pp`'s (spec 21 v2
  §10); O0 / O1 bitwise within `tp` (residency changes nothing); decode at 4k recorded.

### Task 3: the tier on both links

- [ ] **Step 1:** each card's cache, table and misses mirror of its own halves (2 GiB pieces, mapped by that card
  only); the profile shared; swaps per card by 21v2a's policy, the two cards' placements kept identical (one decision,
  both copies) so both halves of an expert are resident together.
- **Gate:** O1 / O2 bitwise; both links streaming at once measured against 21v2a Task 5's both-cards rate; **decode
  ≥ 50 t/s** at 4k, one request, MTP off.

### Task 4: prefill

- [ ] **Step 1:** 21v2a's two prefill modes per card (routed misses only below the crossover, the staging ring above),
  both links filling their rings at once; the TP prefill's all-reduces as Qwen3.8's (17c).
- **Gate:** the count rule against the f32 reference (21v2a Task 7); **prefill ≥ 1000 t/s at 4k**; 32k and 128k
  recorded.

### Task 5: serving and the speed record

- [ ] **Step 1:** `b70-serve` with `--parallel auto` choosing `tp` when faster; MTP (the per-step choice), the prefix
  cache and slots as on `pp`; the tool-call test; passkey; the rows in docs/BENCHMARKS.md and docs/18-running-models.md.
- **Gate:** tool-call test: every first difference passes the f32 judge; passkey passes; served decode ≥ 50 t/s;
  decode with MTP on prose and code recorded against Strata's 66.

### Task 6: past 66 t/s - the decode levers, each on its own measurement

In order of the measured gap (spec 21 v2 §7): the launch count (fuse the gated residual's three launches and QSA's
five, after measuring Flash-Next's fraction of roofline); union-aware verify MoE; a fused QSA decode kernel and int8
KV on the QSA layers (OpenVINO `3b0ee89623` as the reference); int8 hyper-connections converted at load (approved 2026-11-12, f32-judged); the int8 DPAS grouped prefill GEMM and strip-ordered work-groups. If the gates are missed, the mixed
3/4-bit form is the next lever (spec 21 v2 §0.6), brought to the operator with its numbers.

- **Gate (each lever):** its measured gain at 4k (interleaved pairs), the f32 judge passing, bitwise where it changes
  no arithmetic. **The plan's goal:** decode > 66 t/s, one request, MTP off or on (said which), at int4 g64.
