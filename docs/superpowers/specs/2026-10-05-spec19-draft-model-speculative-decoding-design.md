# Spec 19 - speculative decoding with a draft model

**Status:** design, 2026-10-05, for operator review. Open decisions are marked **(decide)**.

**Order:** independent of specs 14-18 for Qwen3.8 (it needs spec 8's verify and commit, which are
merged). Ornith's drafter comes after spec 15c; K2-Horizon after spec 18b (§3, §8).

Every number is **measured** unless marked **derived**, **estimated** or **published**. Published
numbers come from the model cards cited and were measured on other hardware; this spec uses them
only to decide whether P0 is worth running.

## 1. Why

Spec 8's MTP head is the speculative decoder we have, and only Qwen-family checkpoints ship one.
K2-Horizon (spec 18) has none. Separately trained drafters now exist for the models we serve,
including the one we serve most:

| drafter | target | kind | layers x width | block | target taps | attention | size (bf16) | licence |
|---|---|---|---|---|---|---|---|---|
| `z-lab/Qwen3.8-27B-DFlash2` (mirror of `incoai/...`) | Qwen3.8-27B | DFlash 2 | 5 x 5120, FFN 17408, 32 q / 8 kv heads, hd 128 | 8 | layers 5, 19, 33, 47, 61 of 64 | all sliding, window 2048, non-causal | 3.85 GB | Apache-2.0 |
| `ornith-ai/Ornith-1.5-35B-A3B-DFlash` | Ornith-1.5-35B-A3B (spec 15) | DFlash | 6 x 2048, FFN 6144, 32 / 8 heads, hd 128 | 16 | 8 of 40 | 5 sliding + 1 full | 0.77 GB | MIT |
| `z-lab/Qwen3.6-35B-A3B-DFlash` | Qwen3.6-35B-A3B (spec 15's architecture) | DFlash | not inspected | | | | | |
| `IFM/K2-Horizon-3.7B` | K2-Horizon (spec 18) | plain autoregressive model | 36 x 2560 | - | - | causal | 10.1 GB | - |

No DFlash drafter exists for K2-Horizon. `K2-Horizon-3.7B` has a byte-identical vocabulary:
250,000 base ids and 626 added tokens, all with the same ids as the 36B's (`tokenizer.json`
compared). `K2-Horizon-0.9B` has a 64,256-id vocabulary and cannot draft for it.

**Published acceptance** (`Qwen3.8-27B-DFlash2` model card: SGLang, one H200, 7 drafts per
verify, temperature 1.0, top-p 0.95, top-k 20; tokens per verify step, bonus included):

| task | Qwen3.8's own MTP, 7 drafts | DFlash 2 |
|---|---|---|
| HumanEval | 3.91 | 4.39 |
| MBPP | 3.99 | 4.79 |
| MT-Bench | 3.74 | 4.10 |

For our target workload (code, tool calls and the reasoning prose between them), the drafter
proposes more accepted tokens per verify than the MTP head at the same depth. Its draft cost is
also flat in K (§2).

## 2. How DFlash drafts (reference semantics)

Sources:
- vLLM's implementation: `vllm/model_executor/models/qwen3_dflash.py` and `qwen3_dflash2.py`;
  `vllm/v1/spec_decode/dflash.py`; `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py`. All are
  Apache-2.0, and DFlash 2 is from vllm-project/vllm PR 52816.
- The `z-lab/dflash` reference.

A port of the semantics is ours to write; nothing is copied without its licence.

**State.** For every committed position p the drafter holds per-layer K/V derived from the target,
not from tokens:
1. **Taps.** The target's residual stream after each tap layer i (vLLM converts each
   `target_layer_id` i to i+1, i.e. the input of layer i+1), at position p. For Qwen3.8 that is
   5 x 5120.
2. **Context projection.** The taps are concatenated (25,600), then go through `fc` (25,600 → 5120,
   no bias) and `hidden_norm` (RMSNorm).
3. **Per-layer K/V.** Each drafter layer then applies its own k/v projection, `k_norm` and RoPE at
   p (theta 1e7, default rope) to that vector. The result is written to the drafter's KV cache.

Sliding layers only ever read the last `window` positions, so for Qwen3.8's drafter the state is a
2048-position ring:
- size: 5 layers x 2 (K, V) x 8 kv heads x 128 x 2 B = **20 KB per position, 40 MB** (derived);
- context length does not grow it.

**One draft pass per block.**
- **Query rows.** The block has 1 + K rows: [the newest committed token (the target's last
  sample, which has no target hidden state yet), mask x K]. Their embeddings come from the
  **target's** embedding table; the mask id is `dflash_config.mask_token_id`, 248070 for Qwen3.8.
  A checkpoint may ship a separate `mask_embedding.pt` that replaces that row.
- **Layers.** Pre-norm Qwen3 decoder layers whose attention reads the ring (positions < p) plus
  the block's own K/V. Attention inside the block is **non-causal** (`is_causal: false`).
- **Output.** Final norm, then the **target's** `lm_head` on the K mask rows. The drafter ships
  neither embedding nor head.
- **Cost.** One pass drafts all K tokens at once, so draft cost is nearly flat in K. An MTP head
  costs K sequential steps.

**What DFlash 2 adds:**
- **Grouped convolutions.** Two per layer (`attention_conv` around attention, `mlp_conv` around
  the MLP), 2 taps, groups of 16 channels. `kernel_projection` (5120 → 2 x 2 x 320) predicts the
  per-row coefficients, which are added to a learned base. The tap reaches the previous row of the
  block only (`row % (1 + K) >= tap`). Late rows thus see early ones; this is the model card's fix
  for drafts that decay toward the end of the block.
- **Candidate selector.** The head keeps the top 16 ids per mask row, with their logits ("unary").
  A rank-256 bilinear edge score is added: `unary[c] + sum_r pred[prev][r] * h[r] * succ[c][r]`.
  - `pred` and `succ` are [vocab][256] codebooks (127 M parameters together).
  - `h = hidden_projection(row)`.
  - `prev` is the candidate chosen at the row before, and the newest token for the first row.
- **The walk.** A sequential walk picks one id per row: argmax for greedy, Gumbel-argmax at the
  request's temperature for sampling. The draft distribution q at each row is the softmax of those
  16 scores; every other id has q = 0.

DFlash (v1) is DFlash 2 without the convolutions and the selector: each mask row samples from its
own logits. **One engine covers both**, switched by the checkpoint's config.

## 3. Approaches

**A. A DFlash drafter engine (recommended).** For Qwen3.8 now, Ornith after spec 15. The per-step
cost (derived, in units of the int8-head plain step of 32.65 ms, spec 8 §10) is listed below.
- **Drafter weights**, read once per step (5 layers, `fc` and the selector's projections; the
  codebooks are gathered, not streamed):
  - bf16: ~3.6 GB → ~6.3 ms, **0.19**;
  - int8 per-channel, quantised at load: ~1.8 GB → **0.10**.
- **The target head at K rows:** int8 1.27 GB → ~2.3 ms, **0.07** at any K ≤ 8.
- **Context projection** of the j + 1 newly committed rows: included in the weight read above.
- **Draft total: ~0.17 (int8 drafter) to ~0.26 (bf16), whatever K.** MTP's is 0.13 / 0.26 / 0.38 at
  K = 1 / 2 / 3.

The verify grows with K. Spec 8 measured M = 1..4 at 1.00 / 1.17 / 1.52 / 1.74 (bf16 head); M = 5..8
is unmeasured, and a linear extension gives **~2.6-3.1 at M = 8** (estimated). Projected speed-up
over plain decode, with the published acceptance (HumanEval 4.39 at K = 7):
- **4.39 / (2.6 + 0.17) = 1.58x** to **4.39 / (3.1 + 0.26) = 1.31x** (estimated);
- `--mtp auto`, from spec 8 §10's cost table at the measured acceptances: ~1.10x on prose (alpha
  0.44, K = 1) and ~1.76x on tool-call text (alpha 0.97, K = 3) (derived).

So the drafter's gain lies in prose, reasoning and code, where MTP's per-draft acceptance is low
(0.44 on prose). **Verify cost at M = 5..8 decides it**, and P0 measures it first. A cheaper M-row
int4 GEMV (spec 8 A7's lever) helps both MTP and the drafter.

**B. A small autoregressive draft model (not recommended for K2).** K2-Horizon-3.7B is
vocabulary-compatible, but at int4 (~1.3 GB of layers + an int8 head of 0.64 GB, ~3.4 ms per
token, derived) each draft token costs **~0.64 of K2's plain step** (spec 18's roofline ~5.3 ms).
- **Ordinary acceptance (alpha 0.8, K = 3):** 2.95 ids for 1.8 (verify, derived from Qwen3.8's
  M = 4) + 1.9 (drafts) = 3.7 steps, **0.8x**, a loss.
- **Near-perfect acceptance (0.97):** 1.03x.

Two things make it worse for a MoE target at batch 1:
- **The verify rows touch more experts.** M rows x top-8 of 100 select a union of experts, so the
  MoE share of the verify grows faster than for a dense target.
- **A3B / A4B targets have cheap plain steps** to begin with.

Recorded here so it is not re-proposed without a cheaper drafter.

**C. A model-free proposer: prompt lookup / suffix matching (proposed as stage 19e, decide).**
- **How it works.** Drafts are the continuation of the longest earlier match of the last n ids,
  searched in the request's own context (and optionally in earlier requests of the session).
- **Cost.** No weights are read, so cost is the verify alone.
- **Fit to agentic coding.** Agentic coding repeats long spans verbatim (file contents echoed into
  edits, tool output quoted back, unchanged code around a change), which is where this proposer
  accepts most and where MTP and drafters already do well.
- **K2.** The only speculative option K2 has without training a drafter.
- **Gate.** Its value is measured on the opencode recording (§6, D5), not assumed.

## 4. The decisions

1. **(decide) Scope:**
   - proposed: **A for Qwen3.8** (19a-19d), then A for Ornith after spec 15c, then **C for every
     model** (19e), K2 included;
   - B rejected on the arithmetic above.
2. **(decide) Drafter precision:**
   - proposed: **int8 per-channel quantised at load** (RTN, spec 9's `gemv_i8w` form) for every
     drafter linear;
   - P0 compares its acceptance with bf16. Drafter numerics change acceptance only, never the
     output (§6, D2);
   - a published int4 W4A16 drafter (`syvai/Qwen3.8-27B-DFlash2-W4A16`) is a P0 arm: it is half
     the bytes again, if its acceptance holds.
3. **The proposer is one per server:** `--spec mtp|dflash|lookup|off`.
   - `--spec dflash` takes `--draft-model PATH`; the MTP head is then not loaded.
   - `--mtp K` and `--mtp auto` keep their meaning as spellings of `--spec mtp`.
4. **K is per request, from spec 8 §10's policy.**
   - The cost table gains a draft row that is flat in K: `--spec-cost "verify=...;draft=d,d,d,..."`.
   - `--spec-max` bounds K (default 7 for DFlash 2's block of 8; Ornith's block of 16 is capped at
     7 too, because spec 8's GDN slots grow by 151 MB per row).
   - Running at K < block - 1 is legal (the convolutions look back one row).
5. **(decide) Prefix cache (spec 7):** a hit must restore the drafter's ring. Three options:
   - **Store each block's `fc` output plane** (10 KB per position, +16 % over Qwen3.8's 64 KB of
     KV per position) and re-project K/V on restore; proposed.
   - Store the projected ring rows (20 KB, +31 %).
   - Start cold after a hit (no extra memory; acceptance dips until the ring refills from the
     uncached suffix and generated tokens).
6. **A draft-only vocabulary (lever, P0 decides):**
   - **What it is.** The drafter's candidate head reads the full 248,320-row target head. Restricting
     *drafting* to a subset V′ (the most frequent ids on the golden, A4 and opencode corpora, as
     EAGLE-3's draft vocabulary does) cuts that read in proportion.
   - **Output is unchanged.** Verify stays full-vocabulary and bitwise (§6, D3).
   - **The only loss.** Acceptance drops when the target's token lies outside V′; P0 measures that
     rate.
   - **Scope.** The same lever applies to the MTP head's draft pass.

## 5. Design (approach A)

**Per iteration** (one request, K drafts):
1. **Draft list.**
   1. Embed [t_n, mask x K].
   2. Run the drafter's layers against the ring and the block.
   3. Run the target head on rows 1..K, then top-16 and the selector walk.
   4. Write K ids and, for sampling, their 16 (id, score) pairs per row to the readback.
2. **Verify list (spec 8's).** M = K + 1 rows [t_n, d_1..d_K] through the target, bitwise equal per
   row to M = 1 (spec 8 M2); it also **writes the taps** of every row (M x 5 x 5120 bf16, ≤ 400 KB).
3. **Host acceptance (spec 8's `spec_accept`).** Greedy: the longest prefix where the target's
   argmax equals the draft. Sampled: spec 8's lossless rule with the drafter's sparse q. The result
   is j accepted plus one target sample.
4. **Commit list.**
   1. Spec 8's commit: the Control GDN slot index and the KV length.
   2. **The context projection.** For rows 0..j (the positions now committed), `fc` →
      `hidden_norm` → per drafter layer k/v → `k_norm` → RoPE, written to ring slots `p % 2048`.
      The row count is read from Control (j + 1), so the list is captured once.

Every list is captured once and replayed. The host decides only j, exactly as spec 8 does.

**Prefill.**
- Every chunk writes the taps for its rows that fall within the window, i.e. the last 2048 prompt
  positions (a full-attention drafter layer, as in Ornith's, needs all positions).
- A projection kernel then fills the ring from them.
- Chunking does not change the result: the projection is per position.

**Memory (Qwen3.8, derived):**
- drafter: 1.8 GB (int8) or 3.6 GB (bf16);
- ring: 40 MB;
- taps: 0.4 MB;
- spec 8's GDN slots at K = 7: 7 x 151 MB = 1.06 GB (+604 MB over K = 3);
- the MTP head (0.85 GB weights, 0.52 GB buffers) is not loaded.

**Batching (spec 13):** one ring per slot; the draft list batches slots like the verify does.
**PP / TP (specs 16 / 17):**
- the drafter lives on the card that holds the `lm_head`;
- taps from the other card's layers travel with the existing hand-off (M x 5120 bf16 per tap
  layer).

**Kernels.**
- Drafter linears through spec 9's int8 GEMV generalised to M ≤ 8 rows (or bf16 GEMV for the bf16
  arm).
- Drafter attention: spec 10's v2 decode attention with a non-causal block term and a ring
  addressing mode.
- The DFlash 2 convolution and selector walk: small new kernels.
- Top-16 over the head's rows: a new reduction; spec 9's argmax kernel generalised.
- Only the verify path touches the target's kernels: `gemv.cl` already takes M in [1, 8]; the
  M = 5..8 variants are compiled for this spec.

## 6. Correctness gates

- **D0. Nothing changes without `--spec dflash`.** Every merged model's kernel binaries are
  checksum-identical to main's, and the full suite is green.
- **D1. The drafter matches a reference.** A CPU PyTorch port of §2 (Docker on the Mac or the box
  CPU), fed the **engine's own dumped taps**, gives per-row candidate ids and selector scores.
  - the engine's top-16 sets are equal (tie-aware, as the golden gate);
  - scores reach cosine ≥ 0.9999 for the bf16 arm and ≥ 0.999 for int8;
  - walk ids are equal at temperature 0.
- **D2. Greedy is lossless.** `--spec dflash` greedy output equals `--spec off` token for token on
  the golden set and A4, whatever K sequence the policy picks (spec 8 M3).
- **D3. Verify rows are bitwise.** M = 5..8 verify rows equal M = 1 bitwise, logits and GDN slots
  (spec 8 M2 extended).
- **D4. Sampling is exact.** Host tests of the acceptance rule with sparse q:
  - the empirical distribution of 10^6 seeded samples matches the target's within the spec 8 M4
    bars;
  - q = 0 outside the candidates never produces a draft outside them.
- **D5. Speed.** Tokens/s by `b70-serve` with llama-benchy at the operator's flags and on the
  opencode recording, for `--spec off`, `--mtp auto` and `--spec dflash` (auto K):
  - **`dflash` becomes the Qwen3.8 default only if it beats `--mtp auto` by ≥ 10 % on the
    opencode recording** and loses nowhere by more than 3 %;
  - otherwise it ships as an option.

## 7. Stages

- **19a. P0, no new kernels.** Ordered so the cheapest question is answered first:
  1. **Teacher-forced acceptance.** Our engine prefills [prompt + its own greedy output] and dumps
     the taps (a debug flag). The CPU reference drafter then drafts at every position from those
     taps. Its drafts against the actual next greedy ids give the greedy acceptance-length
     distribution at K = 1..7, exactly as a live run would see it (accepted greedy ids are the
     plain-decode ids). Measured on the golden prompts, A4 (tool calls) and a prose / reasoning
     set:
     - for bf16, int8 and the W4A16 drafter;
     - for the draft-only vocabulary at V′ = 32k / 64k / 128k.
  2. **Verify cost at M = 5..8** with `probe_mtp_steps` extended (interleaved pairs, int8 head).
  3. **The projection.** Expected tokens/s per corpus at the best K, against `--mtp auto`'s.
  - **Stopping rule:** if the projection does not beat `--mtp auto` by ≥ 10 % on the coding and
    reasoning sets, stop and record. Spec 8's A7 M-row GEMV lever then comes first, and P0 is
    re-run after it.
- **19b. The drafter.** Loader (`DFlash2DraftModel` / `DFlashDraftModel` configs, the shared
  embedding and head, the optional mask embedding), kernels (§5), the draft list. D1.
- **19c. The engine loop.** Taps in the verify list, the commit-side projection, the prefill tail,
  host acceptance with sparse q, `--spec dflash`, the adaptive-K cost row. D0, D2, D3, D4.
- **19d. Serving and the record.** Prefix-cache plane (decision 5), batching hook, D5,
  BENCHMARKS rows, the spec amendment.
- **19e (decide). Prompt-lookup proposer.** Model-agnostic, host-side matching over the request's
  ids, the same verify / commit path; K2 after spec 18b. D2 (per model), D5 on the opencode
  recording.
- **Ornith's DFlash (v1)** after spec 15c, as 19b-19d with the selector and convolutions off and
  one full-attention layer (needs every position's taps); its 16-row block capped by decision 4.

## 8. Out of scope

- Training or distilling a drafter (including one for K2-Horizon).
- Tree verification (several candidate paths per verify). DFlash 2's selector already walks one
  path; trees multiply the verify rows, which is the cost this spec is bounded by.
- Approach B for any model, unless a much cheaper compatible drafter appears.
- Multimodal inputs.

## 9. Amendment: 19e on the host (2026-10-05)

Plan 19e Task 1 and the host half of Task 2, written on the Mac (branch
`spec19e-prompt-lookup`). Numbers: `docs/probe-prompt-lookup-2026-10-05.md`.

- **Task 1, on A4 only.** No request log exists, so the opencode recording (the deciding
  corpus, D5) is still to come. On the A4 tool-call set (34 prompts, 2613 output ids, greedy,
  the int8-head cost table; M = 5..8 verify costs estimated):
  - lookup alone: **1.44x over plain** at n = 3, K <= 3 with spec 8 §10's policy and a free
    draft (fixed K = 3: 3.10 tokens per verify, 1.45x); K > 3 does not pay at the estimated
    verify costs;
  - against `--mtp auto` projected at A11's 0.92 / 0.87 / 0.82 (1.556x): **0.92x**;
  - combined (lookup when its match >= 6, else MTP K = 3): **1.058x** of `--mtp auto` - below
    §6 D5's +10 % and an upper bound (it assumes MTP's average acceptance where lookup has no
    match).
  - So: for K2 (no MTP head, no drafter) lookup is the speculative option and is worth its
    verify lists; for Qwen3.8 it does not replace MTP on A4, and `mtp+lookup` waits for the
    recording.
- **The matcher** (`server::PromptLookup`): the longest earlier match of the request's ids,
  capped at 64, the most recent on ties, drafts copied forward (overlapping a repeat). Hash
  chains at key lengths 2 / 4 / 8 / 16 / 32 / 64 (zlib's structure): O(1) amortised per
  appended id (~0.23 us), a bounded walk per proposal (~8 us at 262144 ids), ~34 MB at
  262144 ids; exact against a brute force whenever its walk caps (64 candidates, 1024 steps
  per key length) are not reached, which on A4 is every position. A session's next request
  reuses it from the common prefix (`truncate`).
- **Lossless sampling with a point-mass proposal** (`server::accept_point_mass`): accept d
  with probability p(d); on a rejection sample p without d, renormalised. 1e6 seeded draws
  match p (chi-square p-values 0.09-0.78, D4's bar).
- **Serving.** `b70-serve --spec off|mtp|lookup` (decision 3: `mtp` = `--mtp auto` unless
  `--mtp` is given; `--spec` unset = today), `--spec-min-match N` (default 3), `--spec-max K`
  (1..3), `--spec-cost`, `--spec-history N` (match the last N requests' generated ids).
  `EngineIface::verify_k()` / `step_drafts(sampling, propose)` take external drafts; the
  server's matcher holds exactly the prompt and the ids consumed. Qwen3.8 verifies through
  the MTP verify lists (the head is loaded for them; `Engine::verify` reads the drafts from
  `cur_token[1..k]`, which `EngineAdapter::step_drafts` writes) - unvalidated on the card.
  Mock-tested: responses byte-identical to plain over EOS, stop, max_tokens and tool calls
  inside accepted runs.
