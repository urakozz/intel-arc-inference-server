# Spec 9 - a quantised `lm_head`, made at load from the bf16 checkpoint

**Status:** design, 2026-09-28, for operator review.

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

## 2. The decision (proposed)

**Quantise `lm_head` in the loader, from the checkpoint's bf16 tensor**, the
way spec 5 builds its int8 column scales at load: no file on disk changes,
`--lm-head bf16|int8|int4` picks the form, and the choice is gated on
accuracy by P0 and the gates below.

Candidates:

| form | bytes | decode saving (derived at 590 GB/s) | kernel |
|---|---:|---:|---|
| bf16 (today) | 2.543 GB | - | `gemv_bf16` |
| int8, one scale per row (W8A16) | 1.272 GB | ~2.15 ms | **new**: an int8 variant of `gemv_bf16` |
| int4 g64, RTN, sym (as the old checkpoint) | 0.675 GB | 3.05-3.20 ms (measured) | existing `gemv`, layout 1 |
| int4 g64 with an error-minimising rounding (GPTQ-style, one pass on calibration hiddens) | 0.675 GB | same | existing `gemv` |

The default is the smallest form that passes every gate in §4; ties go to the
simpler one. P0 decides which rows are worth building; the int4 RTN row needs
no kernel, so it is measured end to end first.

## 3. Design

- **Loader:** `loader::load(..., LmHeadForm)` reads the bf16 `lm_head`,
  quantises on the host (or on the card with a one-off kernel if the host path
  takes > 10 s: 1.27 G values), repacks to layout 1 for int4 (the one repack
  `load_linear` implements) or to the int8 layout the new kernel reads, and
  frees the bf16 copy. VRAM drops by 1.27 GB (int8) or 1.87 GB (int4), derived.
  The load report gains a line with the form, bytes and quantisation time.
- **Capture:** unchanged for int4 (`Qwen35::lm_head(kind)` already routes it).
  int8 adds one kernel variant and one `WeightKind`.
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

- **P0, first (CPU, then card).** For each candidate: L1 and L2 on the CPU
  from dumped final hidden states (`tools/oracle`); the int4 RTN row end to end
  on the card (no kernel work); load-time cost.
- **H1.** Decode at 4k depth with the chosen form: >= 1.07x (int8) or
  >= 1.10x (int4) plain decode (derived from §2's savings at 33.9 ms/step).
- **H2.** MTP draft step time, recorded; spec 8's derived speedup recomputed.
- **H3.** Prefill unchanged within 1 %.

## 6. Stages

- **9a, probe:** P0; choose the form (or stop: if no quantised form passes
  L1 and L2, record and keep bf16).
- **9b, build:** the loader path, the int8 kernel only if chosen, the CLI
  flags, L1-L4, H1-H3, the record (BENCHMARKS rows with the head form named).

## 7. Out of scope

- Vocabulary pruning for drafts (a separate spec 8 follow-up).
- Quantising `embed_tokens` (gathered, about zero traffic).
- A new checkpoint on disk.
