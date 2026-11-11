# Spec 19 v2 - DFlash for Qwen3.8-27B, made fast

**Status:** design, 2026-11-11, for operator review. Open decisions are marked **(decide)**.

**Continues** spec 19 (`2026-10-05-spec19-draft-model-speculative-decoding-design.md`, implemented: 19a-19f on main).
Spec 19 stays the record of what was built. This spec changes only what the evidence below shows was wrong or is now
the cost.
- **Plan:** `docs/superpowers/plans/2026-11-11-spec19v2-dflash-speed.md`.
- **Open-work:** item 12.
- **Evidence:** `docs/probes/dflash-vs-sly-2026-11-11.md` (the side-by-side with vllm-sly-radiance and saber-engine,
  file:line in both).

**Grades:** **measured** (with its date), **derived**, **estimated**, **published** by others (never a bar by itself).

## 1. Where we start: the findings

### 1.1 What was misread

- **"11 tokens per verify step" elsewhere was batch occupancy.**
  - vllm-sly-radiance's gate number (11.17) is vLLM's `iteration_tokens_total` mean: prompt plus generated tokens per
    engine step, summed over the batch, across concurrency 1-8.
  - Its draft yield is ~5 tokens per verify (1 + 7 x 0.569 acceptance, derived; an older counter run measured 4.20).
  - **Same block as ours:** 8 rows, K = 7, the drafter's trained `block_size` 8. saber-engine serves the same.
  - K = 15 is opt-in at sly and not served by saber.
- **Their step is cheap, not their drafts long.**
  - sly's whole speculative step - an 8-row verify plus the draft pass - is ~33-35 ms. Their published fit is
    T(M) = 29.9 + 0.393 M ms.
  - That is ~1.15 plain steps.
  - saber's draft pass is measured at 4.4 ms (5.5 ms in the stack it ports from).

### 1.2 Our gap, decomposed

| | D5's state (2026-10-27, measured) | today's main (derived from measured parts) | sly (derived) |
|---|---:|---:|---:|
| draft pass | 20.6 ms, 0.70 plain | 8.90 ms, 0.30 (measured 2026-11-04, spec 28d) | ~3-4 ms, ~0.12 |
| verify(8) | 3.17 (gemv) | 1.57 measured before 28c's DPAS attention; ~1.1 derived with it | ~1.03 |
| projection + host | 2.2 ms + 3 fences | 0.40 ms + 3 fences | inside one graph |
| **an iteration at K = 7** | **~3.9 plain steps** | **~1.4-1.9** | **~1.15** |
| tokens per iteration | ~4.1 (K 7, golden + A4, counted past EOS) | not re-measured | ~5.0 (sampled BetterBench) |

**D5's 0.878x was cost per step, not tokens per step.**
- On llama-benchy's prose-like text, DFlash drafted more accepted tokens per verify than MTP (P0: 2.13 at K = 2 against
  MTP's 1.44 at K = 1).
- Its iteration cost 2.29 plain steps against MTP's 1.29: 0.83x derived, against 0.82-0.88x measured.
- The draft pass alone (0.70 against MTP's 0.13) was larger than the whole loss.

**Specs 28b / 28c / 28d have since cut the verify and the draft ~2-3x. Nobody has measured DFlash against MTP since.**

### 1.3 What is still wrong or unknown

1. **The draft pass is 0.30 plain steps, not the 0.17 spec 19 §3 derived.**
   - It reads 2.94 GB: int8 drafter linears 1.665 GB plus the full int8 head 1.27 GB at 7 rows.
   - It runs at ~60 % of the target's bandwidth, through 87 launches in one regular list.
   - sly reads ~1.6 GB (int4 linears, int4 head).
2. **Three submits and three fences an iteration** (draft, verify, commit + projection; spec 19 §11 "Not in 19c").
   The references run the draft inside one graph with no host round trip.
3. **AdaptiveK's DFlash cost row is stale** (`server/adaptive_k.cc:37-45`):
   - draft 0.17 derived, against 0.30 measured;
   - verify(8) 1.56, from before 28c bound DPAS attention at M = 6..8.
   - So it picks K on a wrong model.
4. **In-engine acceptance is not known on a clean workload.**
   - `dflash_gpu_test` reports 4.10 ids an iteration at K 7, but it generates 256 ids past EOS.
   - The teacher-forced reference says A4 7.21 / code 6.15 / prose 2.20.
   - D1 validated the drafter on CPU-dumped taps, never on the engine's own taps in the live loop.
   - A wrong tap or ring path lowers acceptance and never changes an output id.
5. **Precision is not an acceptance problem:**
   - P0: int8 7.10 / bf16 7.09 / W4A16 7.10 at K = 7;
   - D1 int8 equal to bf16 within one draft;
   - both references run int4 or fp8 drafters and an int4 or int2 head.

## 2. The design changes (against spec 19)

Each item names the spec 19 point it replaces. Everything not named stands as spec 19 built it. The draft semantics
(§2), the lossless acceptance (§5 step 3, D2, D4) and the ring in snapshots (decision 5) are among what stands.

**C1. Block and verify width: 8 rows, K <= 7 - kept; spec 19 §13.2 (K up to 15) is parked.**
- Neither reference serves more than 8 rows by default, and the drafter is trained at 8.
- sly's opt-in K = 15 is worth ~10 % on code / json there.
- It costs us 16 GDN slots (+1.2 GB), a deeper conv ring and kernels at M = 9..16.
- Revisit only after C2-C5 meet their bars, and only with the verify(M) fit measured on our card.

**C2. Drafter precision: int4 g64 symmetric linears, quantised at load (replaces decision 2's int8 default).**
- The drafter's eight linear families are q||k||v, o, gate||up, down, fc, the context k||v and the two conv kernel
  projections.
- They bind the target's own int4 g64 small-M GEMM (`gemm_sm`, the verify lists' kernel at M = 8; layout and dequant as
  the target's). **No new GEMM.**
- Bytes: 1.665 GB -> ~0.88 GB (derived), freeing ~0.8 GB of VRAM for context.
- P0 measured the published 4-bit drafter equal to bf16 (7.10 / 7.09). **Our RTN-at-load form is measured on the same
  harness (C6) before it becomes the default.** If RTN loses > 0.05 tokens per verify on any corpus, an AutoRound export
  of the drafter (the one-format rule's path) replaces it.
- int8 stays as a test arm, never a user flag.

**C3. The draft head: the ranked draft vocabulary first, an int4 candidate head with exact re-rank as the fallback
(replaces decision 6's "P0 decides").**
- The head is ~2.8 ms of today's 8.9 ms (measured: the 32k vocabulary's draft list 12.7 against 15.5 ms on gemv).
- **First choice: the 32k ranked draft vocabulary.** It is built (`--draft-vocab 32k`, `--draft-vocab-ids FILE`; the
  default is off). Here it becomes the drafter's default, with the ranking shipped beside the code.
  - It must be ranked **out of sample**: from the opencode recordings, scored on A4 / golden and on a held-out part of
    the recordings.
  - The in-sample P0 number (7.11 vs 7.09) does not count.
- **Fallback (saber's form): an int4 g64 copy of the head, made at load (0.66 GB).**
  - Its top-64 per row is re-scored with the exact int8 head rows (64 x 5120 bytes a row), and the top 16 of those
    feed the selector.
  - A new small gather-dot kernel; used only if the ranked vocabulary loses > 0.05 tokens per verify out of sample.
- Verify always reads the full head: output never depends on this choice.

**C4. One submit an iteration (replaces spec 19 §5's "three lists, the host decides j between them").**
- The commit-side projection of iteration i moves to the front of iteration i + 1's submit.
  - It already reads its row count and position from Control (j + 1 rows at pos - j - 1); the host writes Control before
    the submit, as it does today.
- The submit is **[projection(i), draft(i + 1), verify(i + 1)]**: one queue execution of the three captured lists in
  order, no fence between them.
- The host waits once, reads the verify's ids (and, sampled, the drafter's 16 candidates a row) and accepts.
- The drafts already reach the verify through `Control.cur_token[1..K]` on the device. Nothing the host computes is
  needed between draft and verify.
- Plain steps (K = 0) and the end of a request flush the pending projection exactly as `flush_commit` does today. A
  prefix cache snapshot is taken after that flush, so the ring in it is unchanged.

**C5. The draft list's small kernels (replaces "Work-group reductions only" as a cost assumption, spec 19 §10).**
- Driven by C6's per-launch profile, not by guesses.
- Candidates seen in the code (`runtime/dflash/dflash_capture.cc:66-195`):
  - `df_norm` at grid 1 x 8 (one work-group a row) twice a layer;
  - the `df_conv_prep` / `df_conv_add` pair around every sub-block;
  - two top-k stages over 7 x 248,320 logits;
  - `df_select`, one work-group.
- The fix is fusion (norm into the preceding fold, conv-prepare into the norm) and wider grids. **No change to the
  drafter's arithmetic beyond its order**; the judged gates (spec 28 §11's f32 judge for the drafter) hold.

**C6. Measure before and after (replaces spec 19 §3's derived cost table).**
- A probe times the iteration's parts on the real lists:
  - per-launch events for the draft list;
  - draft, verify(1..8), projection and the host round trip as interleaved pairs at depths 4096 and 32768.
- An engine-taps acceptance harness:
  - the engine's own taps on recorded greedy continuations (A4, golden, the opencode recordings), drafted by the CPU
    reference;
  - set against the engine's own ids per iteration with generation stopped at EOS.
- `MtpCost::dflash` (`server/adaptive_k.cc`) takes the measured verify rows and draft row at each step of the plan that
  moves them.
  - The table stays code constants, not a flag (zero-config).

**C7. Targets** (one card, depth 4096, int8 head, K = 7):

| quantity | spec 19 derived | today | v2 target |
|---|---:|---:|---:|
| draft pass | 0.17 plain steps | 0.30 (8.90 ms) | **<= 0.15 (<= 4.5 ms)** |
| verify(8) | 2.6-3.1 | 1.57 measured / ~1.1 derived | **<= 1.15** |
| projection + host gaps | - | ~0.05 | **<= 0.02 (<= 0.5 ms)** |
| iteration | 2.8-3.4 | ~1.4-1.9 | **<= 1.35** |

- Derived payoff at the target costs with P0's acceptance:
  - code K 7: 6.15 / 1.35 = 4.6x plain, against `--mtp auto`'s ~2.3x;
  - prose K 2: 2.13 / 1.26 = 1.7x, against ~1.23x.
- If C6's engine acceptance sits far below P0's, the payoff shrinks in proportion. That is why C6 runs first.

## 3. Gates

- **G-D2. Greedy identical to `--spec off`** as today: the golden + A4 sets and the opencode replay. Each first
  difference is judged by the f32 judge (spec 28 N6, `spec_judge.h`). Sampled: D4's host tests and the seeded run
  reproducible.
- **G-D0. Nothing changes without `--spec dflash`:** every other list's binaries are checksum-identical, the G0 suite
  green.
- **G-drafter.** The drafter lists judged by the f32 judge (spec 28 §11's form) at each precision change. **The
  acceptance harness is the bar that decides:** tokens per verify on A4, golden code, golden prose and a held-out
  opencode slice within 0.05 of the int8 drafter's.
- **G-accept.** Engine-taps acceptance within 0.15 tokens per verify of the CPU reference on the same anchors (A4,
  code). A larger gap is a defect to root-cause before any speed task, never a bar to widen.
- **G-cost.** Each task's measured cost against C7's row, interleaved pairs, median of rounds, the box idle or the
  grade written next to the number.
- **G-speed (the bar, spec 19 D5 unchanged).** On the opencode recording (`~/b70/logs/opencode-session-int8`, 78
  requests; the other two recordings as checks):
  - `--spec dflash` >= 1.10x `--mtp auto`;
  - no loss > 3 % on A4, golden and llama-benchy tg128 / tg128 @ d32768;
  - prefill loss <= 3 % (the tail's projection).
  - If it passes, `dflash` becomes the one-card Qwen3.8 default whenever the drafter is present - the operator confirms
    (zero-config: the faster path is the default, no new flag).

## 4. Out of scope (recorded)

- **K up to 15 / 16-row verify (C1).**
- **Cactus acceptance (sly's lossy sampled rule):** our output guarantee is lossless.
- **A tree verify (saber's 10-row best-first tree at TP 2):** +7.8 % tokens per step published; it multiplies verify
  rows; revisit after C1.
- **`--parallel tp` and `--max-num-seqs` slots with DFlash.** Refused today. A DFlash slot needs a ring per slot and
  batched draft lists (spec 19 §5 "Batching"). Their own spec after this one meets its bar.
- **The DPAS verify attention's thresholds.** Spec 28c's table is used as built. C6 measures verify(8) through it.
