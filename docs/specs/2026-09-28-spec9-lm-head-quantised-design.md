# Spec 9 - a quantised `lm_head`, made at load from the bf16 checkpoint

**Status:** design, 2026-09-28; **operator ruling the same day: int8, not int4** (§2).

**Order:** after spec 8 (MTP) in the operator's queue; independent of it in
code, but it pays twice under MTP (every draft step reads `lm_head` too).

Every number is **measured** unless marked **derived** or **estimated**.

## 1. Why

`lm_head` is `[248320, 5120]` bf16, **2.543 GB read in full every token**
(`docs/03-models.md`), 16 % of decode's 15.540 GB per token
(`docs/BENCHMARKS.md`). It is the single largest launch in the step.

- Plain decode: an int4 g64 head measured **-3.05 to -3.20 ms per token** and
  **32.22 t/s** against 29.4 (2026-09-04, `docs/BENCHMARKS.md` "The one thing on
  the decode path that was measured three ways"), on a self-quantised RTN
  checkpoint that has since been deleted.
- MTP (spec 8): a draft step is 6.19 ms, of which `lm_head` is 4.39 ms
  (`docs/probe-mtp-2026-09-27.md`); at K = 3 an iteration reads `lm_head` four
  times (derived: three drafts plus the verify's M = 4 rows, which read it once).

The engine already runs an int4 g64 head: `Qwen35::lm_head(kind)` and the
capture pick `gemv` (int4, layout 1, `{K 5120, N 248320, S 1}`) or
`gemv_bf16` from what the loader loaded (`src/model/qwen35.h`,
`src/runtime/capture.cc` "`lm_head` is the one launch ... that depends on the
checkpoint"). It measured at 97.2 % of device bandwidth. What is missing is a
way to get a quantised head **without a new checkpoint**, and an accuracy
gate for it: the old RTN head was never gated against the bf16 head.

**Why it was dropped, and why that does not apply here.** `BENCHMARKS.md`
quotes only byte-matched rows against vLLM, which cannot load a quantised
head. That rule is about a fair comparison. The operator's goal is the
fastest correct engine for agentic sessions (memory:
target-workload-agentic-coding), so the serving default may differ from the
comparison row, as long as both are reported and labelled.

## 2. The decision

**Operator ruling, 2026-09-28: int8, one scale per row (W8A16).** int4 is not pursued:
the head's logits feed both sampling and every MTP draft, and 4 bits per weight is
judged too coarse for them. The int4 rows below stay as the measured history only; P0
gates int8 alone. The MTP head's own weights (`mtp.*`) stay bf16.

**Quantise `lm_head` in the loader, from the checkpoint's bf16 tensor**, the
way spec 5 builds its int8 column scales at load: no file on disk changes,
`--lm-head bf16|int8` picks the form, and the choice is gated on
accuracy by P0 and the gates below.

Candidates:

| form | bytes | decode saving (derived at 590 GB/s) | kernel |
|---|---:|---:|---|
| bf16 (today) | 2.543 GB | - | `gemv_bf16` |
| int8, one scale per row (W8A16) | 1.272 GB | ~2.15 ms | **new**: an int8 variant of `gemv_bf16` |
| int4 g64, RTN, sym (as the old checkpoint) | 0.675 GB | 3.05-3.20 ms (measured) | existing `gemv`, layout 1 |
| int4 g64 with an error-minimising rounding (GPTQ-style, one pass on calibration hiddens) | 0.675 GB | same | existing `gemv` |

int8 becomes the `b70-serve` default if it passes every gate in §4; if it does
not, the head stays bf16 and the result is recorded.

## 3. Design

- **Loader:** `loader::load(..., LmHeadForm)` reads the bf16 `lm_head`,
  quantises on the host (or on the card with a one-off kernel if the host path
  takes > 10 s: 1.27 G values) to int8 rows with one fp32 scale each, in the
  layout the new kernel reads, and frees the bf16 copy. VRAM drops by 1.27 GB
  (derived).
  The load report gains a line with the form, bytes and quantisation time.
- **Capture:** `Qwen35::lm_head(kind)` gains the int8 kind: one new kernel
  variant (`gemv_i8w`, `gemv_bf16`'s structure with int8 weights and a per-row
  scale) and one `WeightKind`.
- **Prefill:** the last row's logits go through the same head (`step_head`
  already reuses the decode binary, `src/runtime/prefill/step.cc`), so the first
  generated token uses the same form as every later one.
- **MTP (spec 8):** the head's draft logits use the same `lm_head`; spec 8's
  acceptance numbers are re-measured with the quantised head (acceptance can
  move either way).
- **CLIs:** `b70-serve --lm-head` defaults to the gated form; `b70-decode`
  defaults to **bf16** so BENCHMARKS rows stay byte-matched with vLLM unless a
  row says otherwise. Every BENCHMARKS row names its head form.

## 4. Correctness gates

- **L1, logits.** On the golden prompts and on the A4 tool-call set, per
  decode position, the quantised head's logits against the bf16 head's on the
  same hidden state: cosine >= 0.9999, argmax equal except at bf16 near-ties
  (the golden rule's tie tolerance), and the **top-20 set** (the checkpoint's
  sampling `top_k`) equal in >= 99 % of positions.
- **L2, sampling.** KL(p_bf16 || p_quant) after temperature 1.0, top-k 20,
  top-p 0.95 (`generation_config.json`): mean <= 1e-3 nats, p99 <= 1e-2
  (the bars are the proposal; P0 reports the distribution first).
- **L3, end to end.** Golden gates on `l0` and `l0-int8` with the quantised
  head (the oracle is the bf16 CPU model, so this measures the head's error
  on top of W4A16's). Gate A4 (tool calls against the unquantised bf16 model):
  no worse than today's 25 / 36 on `l0-int8`.
- **L4.** Replay determinism and every existing registration with
  `--lm-head bf16` unchanged.

## 5. Speed bars

Idle box, device 0, interleaved pairs, median of 3.

- **P0, first (CPU).** int8 per-row: L1 and L2 on the CPU from dumped final
  hidden states (`tools/oracle`); the host quantisation time.
- **H1.** Decode at 4k depth with the int8 head: >= 1.06x plain decode
  (derived: 2.15 ms of 33.9 ms per step, less the kernel's efficiency loss).
- **H2.** MTP draft step time, recorded; spec 8's derived speedup recomputed.
- **H3.** Prefill unchanged within 1 %.

## 6. Stages

- **9a, probe:** P0 (or stop: if int8 fails L1 or L2 on the CPU, record and
  keep bf16).
- **9b, build:** the loader path, the int8 kernel, the CLI
  flags, L1-L4, H1-H3, the record (BENCHMARKS rows with the head form named).

## 7. Out of scope

- Vocabulary pruning for drafts (a separate spec 8 follow-up).
- Quantising `embed_tokens` (gathered, about zero traffic).
- A new checkpoint on disk.

## 8. Amendment, 2026-09-28: L1/L2 redefined (operator ruling)

Plan 9a (`docs/probe-lm-head-2026-09-28.md`) found int8 failing §4's L1 top-20 bar
(92.0 % of positions) and L2's KL mean (+inf) **only** through swaps at the top-20 edge
and the filtered KL going infinite when a near-equal logit moves the top-p cut. A control
that changes nothing but rounding the bf16 head's output to bf16 fails the same bars
worse (86.1 % top-20, 8 support mismatches against int8's 7). The bars as written grade
the bf16 head's own boundary jitter, so the operator redefines them:

- **L1, top-20 set.** A position's top-20 set difference counts against the 99 % bar only
  if the **bf16 head's rank-20 / rank-21 logit gap is >= 0.05**. Swaps across a smaller
  gap are near-ties at the sampling boundary and are not counted. Cosine (>= 0.9999) and
  argmax (equal except bf16 near-ties) are unchanged.
- **L2, sampling.** KL(p_bf16 || p_int8) is measured **without** the top-k / top-p filter,
  over all `vocab_used` ids, at temperature 1.0 **and** 0.6: **mean <= 1e-4 nats and
  p99 <= 1e-3** at each. The **dropped probability mass** - the mass bf16's filter keeps
  on ids int8's filter drops - is recorded, not gated.

Under these bars 9a's numbers pass: every mismatched top-20 row has a rank-20/21 gap of
at most 0.046; the unfiltered KL is mean 1.3e-5 / p99 2.0e-4 at T 1.0 and mean 2.2e-5 /
p99 4.9e-4 at T 0.6; the dropped mass is at most 0.020 (T 1.0) and 0.051 (T 0.6).
**The end-to-end gates L3 (golden gates; A4 >= 25 / 36 on `l0-int8`) are the real test**,
and 9b proceeds to them.
