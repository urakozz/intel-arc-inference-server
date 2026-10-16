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
   - **Ruled in 19d (§12, for the operator to confirm): option 2 at snapshot granularity - the whole projected ring in
     every snapshot**, not per block: a per-block plane or ring cannot be produced for every block (the device keeps
     only the last window's rows: a block finished in decode, or a mid-call block outside the call's last window, has
     none left to store), while a snapshot is taken at its own position and copies the ring as the session holds it.
     +41.9 MB a snapshot (+25 % of the 166.7 MB state). That ring equals an unbroken session's at prompt-end, request-end
     and decode-time snapshots; at a mid-call block end it holds what the call had projected by then - see §12's
     caveat. The cold start stays the behaviour of a snapshot taken without a drafter.
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
  - **Ruled 2026-10-15 (operator): DFlash is built - the stopping rule no longer gates it** ("in vLLM DFlash is a
    monster of performance, we need to support it"). P0's numbers now choose the defaults (drafter format, K, V′) and
    set the expectation; a projection below `--mtp auto` is recorded as a finding for the engine stages to close.
  - **Stopping rule (superseded 2026-10-15):** if the projection does not beat `--mtp auto` by ≥ 10 % on the coding and
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

## 10. 19b as built (2026-10-16)

Branch `spec19b-dflash-drafter`, written on the Mac; box queue row 54 (`tools/box_validate.sh --only r54`). Nothing
of it has run on a card: no `df_*` binary has been compiled (kernel_cmdlines +14 / -0 / ~0) and nothing binds them
without a drafter, so D0 is G0. Plan: `docs/superpowers/plans/2026-10-05-spec19b-dflash-drafter.md` (its status
section lists the rulings and the deviations from the 2026-10-05 plan).

- **The drafter as data** (`model::DFlashDesc`, `src/model/dflash.{h,cc}`): config.json read as `dflash_ref.py`'s
  `config_from_dict` reads it - DFlash 2 or v1 by the config (Review Focus 5), per-layer window / causality as
  `resolve_layer_attention`. Refused by name: a quantization_config (our formats only), another activation, rope type
  or partial rotary, `sample_from_anchor`, no fc, unordered taps, a drafter vocabulary of its own (d2t), attention
  biases, a value scale or sink bias, a conv group that does not divide the hidden size. Held to the target
  (`check_dflash_target`): hidden and vocab_size equal the target's (the embedding and head are shared), every tap
  layer exists, `num_target_layers` matches, the mask id is a row of the shared embedding (Ornith's drafter masks
  with 248077, the first padded row - not a defined token).
- **The loader** (`loader/dflash_repack` host half, `loader/dflash_loader` upload): the checkpoint's names both ways
  after `dflash_ref.py`'s `_canon`; a shipped embedding / head / d2t / mask_hidden refused (the drafter never owns a
  copy - Review Focus 3); bf16 tiled (gemv_bf16) or int8 per-row RTN at load (gemv_i8w, spec 9's quantiser = 19a's
  int8 arm); q || k || v and gate || up fused, the context projection's k || v the q || k || v weight's rows from q_n
  on (no copy); the norms, q / k norms, conv bases fp32; the codebooks bf16 as stored. The mask row is the target's
  embedding row, or a `mask_embedding.safetensors` (vLLM's `.pt` is a pickle: refused naming
  `tools/oracle/dflash_mask_embedding.py`, the converter). `SafetensorsSet` opens a single-file drafter by an opt-in
  flag. Bytes (`loader/dflash_layout.h`, the planner's formula): int8 2,053,464,064 B, bf16 3,849,364,480 B.
- **The kernels** (`src/kernels/dflash/`, names `kernels/dflash_kernels.h`): `df_prep.cl` (embed, norm, the grouped
  two-tap conv prepare / finish, SiLU x up, the block's q / k / v prep, the projection's ring store), `df_attn.cl`
  (the block's 8 rows over the 2048-slot ring plus the block's own K / V, non-causal; **the window is the
  reference's: block row r at p + r sees context >= p + r - 2047**, so row 0 sees 2047 positions and row 7 2040 - the
  plan's "max(0, p - 2048) .. p - 1" was superseded by D1's reference), `df_topk.cl` (the 16 best per head row, two
  fixed stages, ties to the lower id, NaN never, ids >= 248077 masked as argmax.cl masks them; a draft vocabulary's
  compact indices mapped to ids before anything reads them), `df_select.cl` (DFlash 2's walk: greedy, or Gumbel with
  vLLM's murmur3 keying at the request's temperature and 64-bit seed; v1's per-row pick, greedy only). Work-group
  reductions only - every kernel runs on the Mac's GPU too (tools/mac/clrun/dflash_run). Numerics: residual and
  every norm fp32, GEMV inputs bf16 (the reused gemv kernels read bf16 - vLLM runs the drafter in bf16 throughout),
  the ring bf16 (20 KB a position), the block's K / V fp32, FP_CONTRACT off.
- **The lists** (`runtime/dflash/`): `DFlashBuffers` by the one formula (`dflash_sizes`: ring 41,943,040 B, taps
  409,600 B, RoPE 512 B a position, 62.3 MB at max_len 16384 with the full head), `runtime::DrafterPlan` in
  `plan()` / `max_len_that_fits()` (the memory line gains "drafter X GB," only with one; on a 32 GB card with a 17 GB
  target auto goes 190208 -> 156928 with the int8 drafter, derived), `build_dflash_draft` (87 launches: K, the
  walk's temperature and seed in the drafter's control block `DFlashCtl` - one list for every K, rows past K masked;
  the anchor from `Control.pos` / `cur_token[0]`, the drafts into `Control.cur_token[1..K]` where the verify list
  reads them; the readback `DFlashDraftOut`: per row 16 candidates as ids, their head logits, the selector's
  realized scores, the choice) and `build_dflash_project` (8 launches, `proj_n <= 8` rows at `proj_pos` - 19c's
  commit-side projection and prefill tail). The head is the target's at 7 rows: `gemv_i8w` / `gemv_bf16` over the
  full head or the draft vocabulary's compact one (`LoadedModel::draft_vocab`, which `loader::load` still builds only
  with the MTP head - 19c relaxes it with `--spec dflash`).
- **Measured on the Mac:** the host twin `tests/kernels/dflash_ref.h` equals `dflash_ref.py` to 6e-7 on its own
  fixture (rows, top-16 order, scores, walks - two of them sampled; `dflash_ref_test`); the lists on the Mac's GPU
  equal the twin (`dflash_run`, indicative); a draft vocabulary holding every candidate gives the full head's draft
  bitwise (Review Focus 3, on the Mac at 256 ids and in `dflash_draft_test` at 32768 on the card). A preview with
  the real drafter on 19a's real taps (no target head on the Mac: random anchor rows and head) put the bf16 engine's
  final rows at cosine 0.99991 of the fp32 reference (0.99999 with fp32 GEMV inputs and the bf16 ring) - D1's
  scores bar (0.9999) is measured on the box against the real head (r54.d1); if it fails while the sets and walks
  hold, fp32-input GEMV variants are the lever.
- **D1** (r54.d1 / r54.d1_ref): `tools/probe/dflash_d1` drafts at teacher-forced anchors of 19a's tap dumps (the
  engine's own taps arrive with 19c's verify list; the tool takes either) and `tools/oracle/dflash_d1.py` runs
  `dflash_ref.py` on the same taps, anchors and head form, three arms (int8 / int8, bf16 / bf16, int8 + a 32k V').
- **Not in 19b:** int4 g64 drafter weights (no checkpoint of ours; 19a measured equal acceptance), v1 sampling (full
  vocabulary Gumbel: refused by `runtime::dflash::set_draft_params`; df_select's v1 build drafts nothing at T != 0), causal or full-attention drafter layers (Ornith's v1 drafter: its stage), the engine loop,
  `--spec dflash` and the taps (19c).

## 11. 19c as built (2026-10-16)

Branch `spec19c-dflash-engine`, written on the Mac after 19b (itself unvalidated); box queue row 55
(`tools/box_validate.sh --only r55`). Nothing of it has run on a card: 7 new binaries (`gdn_step_slots_M1..4_N8`, the bf16
lm_head at M = 5 / 6 / 8; kernel_cmdlines +7 / -0 / ~0), none bound without `--spec dflash`, so D0 is G0. Plan:
`docs/superpowers/plans/2026-10-05-spec19c-dflash-engine-loop.md` (its "As built" lists the deviations).

- **The Engine's view of a drafter** (`runtime/tap_drafter.h`): `runtime::TapDrafter` - its taps (the layers and the
  [8][taps][hidden] bf16 buffer), the window it reads back, the projection list for (pos, n), a reset, its bytes, and
  the proposer side (`draft(engine, k, T, seed)` -> the 19b readback, `samples()`). `runtime::dflash::DFlashDrafter`
  implements it over 19b's weights, buffers and lists (`runtime/dflash/dflash_drafter.h`: build the drafter, the
  Engine with it, then `attach()` captures the draft list against the Engine's model and Control).
- **Per iteration** (§5, as built): the draft list (anchor `Control.pos` / `cur_token[0]`, drafts into
  `cur_token[1..K]`) -> `Engine::verify(K)` on the **tap verify list** (`runtime::build_verify_taps`: the decode list at
  M = K + 1 rows, gdn_step's SPEC_SLOTS build over 8 GDN slots the Engine owns, and after the input fold of layer
  i + 1 each row's residual - layer i's output, vLLM's aux index i + 1 - copied into the taps; `decode_launches` at
  every M, 5 x M row copies) -> host acceptance (`server::accept_greedy`, or `server::accept_sparse` with the
  drafter's sparse q) -> `Engine::commit(j, t)`: `Control.gdn_live = (live + j) % 8`, pos = n + j + 1, then the
  projection of verify rows 0..j at positions n .. n + j into the ring (replayed before commit returns, so nothing
  rewrites the taps first). Review Focus 1: rows past j are never projected. Review Focus 2: t has no tap; it is the
  next draft's anchor. ingest() and generate() run verify(0) + commit(0) (pos 0 included: no h_{-1} is needed), so
  the ring stays filled across plain and speculative steps.
- **The GDN state at the last accepted token** (the operator's question, against vllm-xpu-kernels 78799fe's per-token
  replay state): the verify list writes the state after EVERY row into its own slot ((live + m) % 8); committing j
  accepted drafts only moves the slot index - no copy, no replay. The rejected rows' slots are never read before a
  later list rewrites them, and their conv-ring positions (pos % 16) are rewritten by the next rows before any row
  reads them (gdn_step.cl's slot disjointness at ring depth 16 >= M + 3). Held by `tools/mac/clrun/gdn_slots_run` on
  the Mac's GPU (the 8-slot build bitwise to the plain kernel; partial acceptance at j = 0 / 3 / 6 / 7 of 7 and 2 of 4
  continues exactly as the plain run; a mutant committing j + 1 is caught) and on the card by `dflash_engine_test` P
  (the live slot against the state after j plain steps, then 64 greedy ids). **Its cost, and a lever (not built):**
  each verify writes M x 151 MB of state (1.2 GB at M = 8, ~2 ms at ~600 GB/s, derived) and the 7 extra slots hold
  1.06 GB. vLLM's RecoverSSM verify reads a read-only state and records per row a replay entry (the delta-rule
  correction [V], the normalised k [K], the decay - ~2.4 MB at M = 8 on Qwen3.8) and commit replays the accepted rows'
  rank-1 updates in one pass (~0.5 ms); that would return ~1.5 ms a verify at M = 8 and 1.06 GB to the KV cache, at
  the price of a commit kernel whose replay must be bitwise to the in-line recurrence (vLLM adds a + dt_bias in fp32
  for the same reason). Read for ideas only; nothing copied.
- **The prefill tail:** every chunk copies its rows inside the drafter's window of the call's end (the last 2048
  positions) at each tap site into a [row][tap][hidden] staging (one 2D region copy per tap, `prefill::Context::
  copy_2d`), then projects them 8 rows a replay (~256 replays for a full window, each reading ~183 MB of fc and k||v
  weights: ~80-100 ms a long prompt, derived; a projection GEMM at M = C is the lever). Chunking does not change a
  position's projection; a split at a multiple of 64 is bitwise (`dflash_engine_test` PF). Refused by name: the
  sycl-tla backend and `B70_PREFILL_REPLAY` (the taps are copies, which a recording cannot hold), a drafter tapping
  the last layer (prefill folds it for the last row only; no published drafter does).
- **Sampling (D4):** q_j = softmax(scores_j / T) over draft row j's 16 candidates (the walk's realized selector
  scores), 0 elsewhere (`server::SparseQ`, `dflash_q`); T is filter_probs' (<= 0 -> 1.0). `accept_sparse` is spec 8's
  rule over it: a draft with q = 0 is a logic error, the residual max(0, p - q) lives on p's filtered support (p is
  whole there off the candidates), and p == q falls back to p. EngineAdapter draws one fresh 64-bit Gumbel key per
  iteration from the request's RNG (then the uniforms, then the correction draw), so no position's noise is reused
  across iterations. `dflash_accept_test` (10^6 seeded draws): chi-square p 0.59 / 0.23 unfiltered, 0.29 / 0.91 at
  top-k 12 / top-p 0.8 / T 0.7; no accepted draft outside the candidates; a mutant residual (p instead of p - q) fails
  at p 0.0000. A v1 drafter (greedy only, 19b) decodes sampled requests plainly.
- **Serving:** `b70-serve --spec dflash [--draft-model PATH] [--draft-format int8|bf16] [--spec-max 1..7]
  [--spec-cost SPEC] [--draft-vocab N]` rides the server's `--mtp auto` loop: `EngineAdapter::step_many` drafts
  through TapDrafter, `server::AdaptiveK` chooses K over `MtpCost::dflash` (verify M = 1..4 the head form's rows,
  M = 5..8 extrapolated at +0.22 a row; the draft row flat in K: drafter 0.10 int8 / 0.19 bf16 + the head's K rows 0.07
  int8 / 0.14 bf16 - all DERIVED until row 29's probes and row 55 measure them; AdaptiveK then picks K 7 at alpha 0.95,
  1 at 0.5, 0 at 0.1). `b70-decode --ids --spec dflash` runs greedy at a fixed K (`--spec-max`). The draft
  vocabulary is built without the MTP head (`DraftVocabSpec::for_drafter`). Refused before the device, by name: `--pp 2`
  (the taps across the hand-off are 19d's), `--kv-cache int8` (kv8.cl is built for M <= 4), `B70_DECODE_ATTN=v1`
  (not built at M = 5..8), `--mtp`, other models.
- **Memory:** the planner counts the engine's side of a drafter - 7 x 150,994,944 B of GDN slots in decode state and,
  prefilling, the 104,857,600 B tap staging in prefill scratch - beside 19b's drafter term; `--max-len auto` on a 32 GB
  card with a 17 GB target: 139520 with the int8 drafter (19b's 156928 did not count these).
- **Prefix cache (Review Focus 5):** the ring is not part of a snapshot; a restored session's ring holds whatever it
  held (zeros after a reset) until prefill and commits refill it - correct, lower acceptance only; decision 5's plane
  is 19d's.
- **Not in 19c:** `--pp 2` and batching (19d), int8 KV and v1 attention at M = 5..8, D5 and the BENCHMARKS rows (19d),
  an engine-side tap dump for D1 (r54.d1 still feeds 19a's CPU dumps; `dflash_engine_test` T1 holds the engine's taps
  to its own M = 1 path), the replay-state lever above, fusing the projection and the next draft into one submit (3
  submits an iteration today: draft, verify, projection).

## 12. 19d as built (2026-10-16)

Branch `spec19d-dflash-serving`, written on the Mac after 19b and 19c (both unvalidated); box queue row 56
(`tools/box_validate.sh --only r56`). Nothing of it has run on a card; it adds no binary (kernel_cmdlines +0 / -0 / ~0)
and nothing changes without `--spec dflash`, so D0 is G0. Plan: `docs/superpowers/plans/2026-10-05-spec19d-dflash-serving.md`
(its "As built" lists the deviations from the 2026-10-05 plan, each with its cost if wrong).

- **Decision 5 as ruled: the ring rides in every snapshot.** `runtime::TapDrafter` gains `ring_bytes` / `save_ring` /
  `load_ring` (the whole ring, ring_k then ring_v, 41,943,040 B for Qwen3.8's drafter) and `key()`; `Engine::state_bytes /
  save_state / load_state` append it after the target's state (and `PipelineEngine`'s, in the same one-card layout, so a
  snapshot moves between `--pp 1` and `--pp 2`). A restore is a copy: the ring after it is bitwise the ring the snapshot
  found (Review Focus 1, by construction). **The caveat, recorded:** a mid-call block end b's snapshot holds the ring as
  the call left it there - the prefill projects only the call's last 2048 positions (19c's tail), so of the window
  [b - 2048, b) every position the call reached but which lies before the call's last 2048 is unprojected (the slot
  holds an older position's row or zero). For a block end at or before the call's end - 2048 that is the WHOLE window:
  such snapshots carry 41.9 MB that buys nothing (a 64k first prompt makes ~30 of them). A restore there equals a cold
  run's ring only when the new tail is at least 2048 long; otherwise it drafts worse until the ring refills (never
  another output). r56.d5_opencode records each request's prefix plan beside its dflash acceptance, so the operator can
  see whether such restores happen and cost; the choices then are to project every position under the hook, to treat
  those restores as cold (and not store their ring), or to keep this. Projecting every position under the hook would cost ~90-130 ms a block
  (~10-13 % of prefill, derived) - the lever if the recording shows such restores matter. The plane per snapshot (the
  normed fc output, 20 MB, re-projected at restore, ~30-50 ms derived) is the lever for the memory.
  **Ruled 2026-10-16 (operator): keep this** (option a) - rings stored as the prefill left them; revisit only if
  r56.d5_opencode's prefix plans show such restores happen and cost acceptance.
- **Store keys carry the drafter (Review Focus 2):** `server::store_root(kv_form, drafter_key)` is the root of every hash
  chain - `kv_form` itself without a drafter (every key as before), a mix of both with one. The key is FNV-1a over the
  drafter's shapes, taps, format and its two norm weights as loaded (`runtime::dflash::drafter_key`).
  `EngineIface::drafter_key` / `BatchEngineIface::drafter_key` (0 = none) feed PrefixSession and PrefixSlots.
- **Batching (spec 13c):** no batched runtime engine exists (13b was never built: the scheduler runs over its mock), so
  the hook is the interface and the scheduler: `BatchEngineIface::spec_k / step_spec / truncate_to(slot, pos)` - one
  draft / verify / accept iteration over every row, each row drafting from its own slot's state (one ring per slot,
  Review Focus 3) - and `Scheduler::spec_round`: K per row from the request's own `AdaptiveK` (or a fixed K), the run's
  ids through the per-id rules, the ids past an EOS / stop / max_tokens handed back before the request-end snapshot.
  `MockBatchEngine`'s speculative mode keeps a ring per slot whose content decides the slot's acceptance.
- **`--spec dflash --pp 2`** (19c refused it): the drafter on device 1 beside lm_head and an embedding replica (2.54 GB,
  `place_stages(..., drafter)`); each device's verify slots for 8 slots (`pp_gdn_spec_bytes(d, gdn, 8)`); the stage tap
  verify lists (`build_stage_verify_taps`: device 0 copies the taps of its sites straight into device 1's tap buffer,
  before its hand-off - so **the copy hand-off only**: `--pipeline-handoff peer` is refused by name); `PipelineEngine::
  draft_tap` (the draft list on device 1, the drafts mirrored to device 0), commit moves gdn_live mod 8 on both and
  replays the projection on device 1; the pipeline prefill taps the call's last window on both devices into a per-chunk
  staging on device 1 and projects a chunk's rows from the host once both devices have finished it (before its block
  hook). `TapDrafter::attach_target / draft_with` let a drafter serve a target that is not a runtime::Engine.
  `PipelineEngineAdapterT`'s DFlash path is EngineAdapter::step_dflash's; the planner's `PpExtras::drafter` counts the
  drafter, the replica, both devices' slots and the stagings (Qwen3.8, int8 drafter, cache on: auto split 36, 262144
  on two cards, derived). b70-decode stays one card.
- **The defaults, chosen from P0 (the 2026-10-15 ruling):** `--draft-format int8` (P0: equal acceptance at half the
  bytes; our int4 g64 drafter is a later lever), `--spec-max 7` (the block; acceptance still rises at 7 on tool calls and
  code, and AdaptiveK lowers K per request on prose), `--draft-vocab off` (the ranked 32k was ranked in-sample; r56.d5_dv
  measures it), `--spec` unset (= `--mtp`'s meaning): `dflash` becomes the Qwen3.8 default only on D5's bar.
- **D5:** prepared, not run - `r56.d5_a4` (A4 + golden through b70-serve: off / `--mtp auto` / dflash / dflash K 3,
  rotated rounds), `r56.d5_dv`, `r56.d5_benchy` (row 41's flags at concurrency 1), `r56.d5_opencode` (the bar: the
  recording, which does not exist yet - the stage SKIPs without OPENCODE_LOG), `r56.d5_pp`. BENCHMARKS "DFlash (spec 19)"
  holds the rows, pending.
- **Tests:** host - `dflash_prefix_test` (restore-and-continue through PrefixSession over a mock whose ring fills as the
  engine's: the ring equals the unbroken session's; 19c's ring-less snapshot caught; the key), `dflash_scheduler_test`,
  `pp_dflash_serve_test`, `pp_serve_plan_test` 7, `dflash_plan_test`'s ring / key rows; box - `dflash_prefix_gpu_test`
  (19c's deferred minor 8: R1 the ring bitwise after a restored tail, the next ids and every iteration's acceptance
  equal; R2 a block-end restore; the caveat counted; a cold-ring control), `dflash_engine_test` SR, `pp_dflash_test`
  (`--pp 2` bitwise one card's).
- **Not in 19d:** Ornith's DFlash (v1) drafter (a full-attention layer and a 16-row block - its stage), a batched runtime
  engine (spec 13b), int8 KV / v1 attention at M = 5..8, the replay-state lever (§11), the measured cost table (row 29).

