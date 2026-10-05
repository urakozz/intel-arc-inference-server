# Prompt-lookup acceptance, offline (spec 19e Task 1), 2026-10-05

Spec: `docs/superpowers/specs/2026-10-05-spec19-draft-model-speculative-decoding-design.md`
(§3 C, §4 decision 1, §9). Plan: `docs/superpowers/plans/2026-10-05-spec19e-prompt-lookup.md`.
Tool: `tools/spec/lookup_accept.py` (tests: `tools/spec/test_lookup_accept.py`). No box: this
is a replay of recorded ids through a host model of the verify loop.

**Caveat first.** The corpus the decision rests on is the **opencode recording**
(`b70-serve --log-requests`, spec 7 P0), which does not exist yet. No request log is in the
repo. The only agentic corpus at hand is the A4 tool-call set: 34 single-turn prompts (two
of the 36 have no reference output) whose outputs are short tool calls (29-192 ids, 2613 in
all). Every number below is A4's and says what lookup does on **one-shot tool calls**, not on
a long agentic session with file contents echoed into edits across turns. Rerun on the
recording: `lookup_accept.py --log <dir> [--seed-session N]`.

## Method

- **Corpus.** Prompt = `tests/golden/toolcall/<name>.ids` (the rendered chat, thinking off).
  Output = the scenario's reference text in `tests/server/toolcall_expected.json` (bf16
  backend first, then l0, then l0-int8), **re-encoded** with Qwen3.8's `tokenizer.json`
  (HF `Qwen/Qwen3.8-27B`, downloaded for this run, not committed), plus `<|im_end|>` for the
  31 complete calls; the 3 incomplete edits stop where the reference run stopped (32 bf16
  references, 2 l0). The re-encoding can differ from the model's own ids where a text has
  two tokenisations.
- **Target.** Greedy, its output the recorded ids. The engine protocol (spec 8 §3.3): the
  pending id is known, the matcher holds prompt + output up to it, the drafts continue it; a
  verify of M = K + 1 rows emits the pending id and the kept drafts; no match >= n is one
  plain step.
- **Matcher.** `src/server/prompt_lookup.h`'s semantics: the longest earlier match of the
  context's suffix, capped at 64 ids, the most recent on ties, overlapping copies allowed.
  The C++ matcher (default caps) gives the **same proposal at all 2613 positions** as the
  Python reference (cross-checked with a scratch harness).
- **Costs** (spec 8 §10, plain-step units): verify M = 1..4 = 1.00 / 1.18 / 1.56 / 1.79
  (int8 head, **derived**); M = 5..8 = 2.10 / 2.40 / 2.71 / 3.01 (**estimated**, the
  M = 2..4 slope extended; spec 19 §3 A's ~2.6-3.1 at M = 8). Lookup's draft cost is 0 (host,
  ~8 us per proposal at 262144 ids against a ~32.65 ms step). MTP drafts 0.13 / 0.26 / 0.38.
- **`--mtp auto`** is projected, not replayed: spec 8 A11's 0.92 / 0.87 / 0.82 read as
  per-depth acceptance gives E(3) = 3.38 ids per (1.79 + 0.38) = **1.556x** plain at K = 3.
  A11 measured those on the golden + A4 prompts; on A4 alone A12 measured 1.43-1.63x.
- **Combined** (Review Focus 5, one proposer per iteration): lookup's drafts when its match
  is >= n, otherwise an MTP iteration at K = 3 whose kept count is drawn from the per-depth
  model (20 seeds). This assumes MTP's average acceptance where lookup has no match, which
  over-rates the combination: copy-like text is where both accept.

## Results (A4, 34 sequences, 66064 prompt ids, 2613 output ids)

Tokens per verify (iterations that verified, bonus included) / speed-up over plain, fixed K:

| n \ K | 1 | 2 | 3 | 4* | 5* | 6* | 7* |
|---|---|---|---|---|---|---|---|
| 2 | 1.75 / 1.371x | 2.30 / 1.364x | 2.76 / 1.413x | 3.04 / 1.351x | 3.41 / 1.328x | 3.59 / 1.257x | 3.71 / 1.186x |
| 3 | 1.81 / 1.346x | 2.58 / 1.409x | **3.10 / 1.450x** | 3.52 / 1.425x | 4.08 / 1.434x | 4.29 / 1.372x | 4.49 / 1.320x |
| 4 | 1.89 / 1.329x | 2.47 / 1.323x | 3.05 / 1.377x | 3.46 / 1.352x | 3.95 / 1.351x | 4.15 / 1.300x | 4.39 / 1.262x |
| 5 | 1.80 / 1.258x | 2.39 / 1.261x | 2.91 / 1.297x | 3.30 / 1.277x | 3.75 / 1.272x | 3.96 / 1.232x | 4.14 / 1.194x |
| 6 | 1.88 / 1.244x | 2.58 / 1.263x | 3.17 / 1.298x | 3.67 / 1.294x | 4.28 / 1.302x | 4.62 / 1.280x | 4.85 / 1.250x |

\* K > 3 needs verify lists at M = 5..8, which the engine does not capture yet; their costs
are estimated.

Share of iterations that verify (a match >= n) and per-draft acceptance, K = 3:

| n | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| verify share | 0.642 | 0.473 | 0.390 | 0.335 | 0.261 |
| accepted / drafted | 0.586 | 0.699 | 0.684 | 0.638 | 0.722 |

`--spec lookup` as built (spec 8 §10's AdaptiveK with free drafts), speed-up over plain and
against projected `--mtp auto`:

| n | K <= 3 | K <= 7* |
|---|---|---|
| 2 | 1.415x (0.909x) | 1.387x (0.892x) |
| 3 | **1.438x (0.924x)** | 1.407x (0.904x) |
| 4 | 1.382x (0.888x) | 1.374x (0.883x) |
| 5 | 1.311x (0.842x) | 1.295x (0.832x) |
| 6 | 1.298x (0.834x) | 1.277x (0.821x) |

Combined with MTP (lookup when its match >= n at K, else MTP K = 3), over plain / over
`--mtp auto`:

| n \ K | 1 | 2 | 3 | 4* | 5* | 6* | 7* |
|---|---|---|---|---|---|---|---|
| 2 | 1.535x / 0.986x | 1.548x / 0.995x | 1.608x / 1.033x | 1.563x / 1.005x | 1.570x / 1.009x | 1.491x / 0.958x | 1.423x / 0.914x |
| 3 | 1.551x / 0.996x | 1.572x / 1.010x | 1.638x / 1.052x | 1.605x / 1.031x | 1.611x / 1.035x | 1.544x / 0.993x | 1.491x / 0.958x |
| 4 | 1.557x / 1.001x | 1.563x / 1.004x | 1.624x / 1.044x | 1.601x / 1.029x | 1.607x / 1.033x | 1.547x / 0.994x | 1.499x / 0.963x |
| 5 | 1.553x / 0.998x | 1.566x / 1.006x | 1.623x / 1.043x | 1.607x / 1.032x | 1.602x / 1.030x | 1.566x / 1.007x | 1.520x / 0.977x |
| 6 | 1.563x / 1.005x | 1.580x / 1.015x | **1.647x / 1.058x** | 1.635x / 1.051x | 1.630x / 1.047x | 1.606x / 1.032x | 1.568x / 1.008x |

By match length (n = 2, K = 3): a longer match keeps more drafts.

| match | 2-3 | 4-7 | 8-15 | 16-31 | 32-63 | 64 |
|---|---|---|---|---|---|---|
| iterations | 347 | 169 | 183 | 81 | 8 | 0 |
| tokens / verify | 2.25 | 3.14 | 3.07 | 3.40 | 3.25 | - |

By scenario (n = 3, K = 3): the edits that quote the file (t3 rename, t4 comment) are where
lookup pays; a search pattern (t1) or a test file (t5) is mostly new text.

| scenario | out ids | tokens / verify | fixed K = 3 | AdaptiveK <= 3 |
|---|---|---|---|---|
| t1 define | 234 | 2.70 | 1.208x | 1.188x |
| t2 explain | 231 | 2.78 | 1.421x | 1.393x |
| t3 rename | 764 | 3.32 | 1.603x | 1.603x |
| t4 comment | 760 | 3.35 | 1.563x | 1.563x |
| t5 test | 432 | 2.72 | 1.256x | 1.228x |
| t6 usage | 192 | 2.88 | 1.384x | 1.370x |

The 64-id cap on the match length changes nothing on A4: caps of 8, 16, 64 and 1024 give the
same numbers to four places.

## What it says

1. **Alone, lookup gives ~1.44x over plain on A4** (n = 3, K <= 3, the policy as built):
   the only speculative option a model without an MTP head or drafter has (K2-Horizon, spec
   18), so for K2 it is worth building (after 18b), pending the recording.
2. **Alone it does not beat `--mtp auto` on A4** (0.92x of the projection). For Qwen3.8 it
   is not a replacement for MTP on this corpus.
3. **Combined, +3-6 % over `--mtp auto`** (best n = 6, K = 3: 1.058x) - below spec 19's
   +10 % bar for a default and an over-estimate (above). The plan's condition for
   `--spec mtp+lookup` ("if Task 1 shows the combination pays") is **not met on A4**; the
   recording decides, where long verbatim echoes across turns favour lookup more than A4's
   short calls do.
4. **K > 3 does not pay on A4** with the estimated M = 5..8 verify costs: tokens per verify
   rise (3.10 -> 4.49 at n = 3) but the verify cost rises faster. Long echoes (the recording)
   may differ; the M = 5..8 costs are spec 19 P0's to measure.
5. **n = 3** is the best minimum match alone (n = 2 drafts on too many 2-id coincidences:
   0.59 acceptance); combined with MTP a stricter n (6) is better, because MTP covers the
   positions lookup gives up. `b70-serve`'s default is n = 3.
