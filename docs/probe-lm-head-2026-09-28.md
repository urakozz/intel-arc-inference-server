# Probe: an int8 `lm_head` against the bf16 head (spec 9 P0), 2026-09-28

**Verdict: under the stopping rule, int8 per-row (W8A16) FAILS spec 9 §4 as written.**
Two bars fail: the L1 top-20 set (92.0 % of positions, bar 99 %) and the L2 KL mean
(+inf, because 7 of 1140 positions keep a token under bf16 that int8 drops). Cosine,
argmax and the KL p99 pass. Plan 9a §6 says stop, so the head stays bf16 until the
operator rules.

**But the two failing bars are below bf16's own noise floor.** The control is the same
fp32 bf16-head logits with nothing changed except that the output is rounded to bf16,
which is exactly what the golden oracle's logits hold. That control fails the same two
bars, by more on top-20 (86.1 %) and by the same count on support (8 of 1140 at T = 1.0,
and at T = 0.6 the same 9 as int8). Every int8 top-20 difference is **one id swapped at
rank 19 or 20**, and the reference's own logit gap there is at most 0.046, median 0.010.
The operator has to decide whether the bars were meant to grade that boundary (§5 below).

Every number here is **measured** unless marked **derived**.

## 1. What was run

- **Heads.** The bf16 head is `logits = fp32(h) @ fp32(W)^T`. The int8 head is
  `(fp32(h) @ fp32(q)^T) * s`, where `q = rne(W / s)` is clamped to [-127, 127] and
  `s = max|W_row| / 127` is an fp32 scale per row: symmetric, one scale per row, and the
  hidden is not quantised. `W` is the checkpoint's own bf16 `lm_head.weight`
  `[248320, 5120]` (`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`). Both heads
  accumulate in fp32.
- **Hidden.** The hidden is **after** `model.norm`, and the dump keeps that input to
  `lm_head` as bf16. It comes from the dequantised W4A16 CPU oracle
  (`mtp_ref.build_main` / `main_forward`, the golden model), teacher-forced over
  ctx + continuation.
  - **Review Focus 1 check:** the bf16 head applied to the dumped hidden, against the
    model's own logits on the same rows, gives **cosine min 0.99999858** over all 24
    sources (bar 0.99999). The residual is the model's bf16 output rounding.
- **Positions.** The positions are every decision row, from len(ctx)-1 through the last
  continuation token (len(cont)+1 rows per source).
  - **golden:** the three golden prompts, each with its 32 greedy golden tokens. That is
    99 rows.
  - **toolcall:** 21 A4 scenarios, each with the bf16 oracle's own output
    (`~/b70-toolcall/toolcall-out/*.bf16.ids`). There is one scenario per task type on
    `box` (t1-t6), plus t1/t2/t6 on the other five files. That is 1041 rows.
  - The 15 long t3/t4/t5 scenarios on non-`box` files were left out, because their CPU
    forward costs ~30 min each (t3/t4/t5 on `box`: 1768-2034 s).
- **Vocabulary.** Every metric covers ids < 248077 only (`vocab_used`; the container's
  tokenizer length agrees).
- **Sampling.** The filter is the engine's `sample()` (`src/cli/serve_adapters.h`):
  top-k by logit, then softmax at temperature, then a top-p cut after the first prefix
  whose mass is >= top_p. The values come from `generation_config.json`
  (T 1.0, top-k 20, top-p 0.95), with a second run at T 0.6.
  - `KL(p_bf16 || p_int8)` is taken over the filtered distributions. It is +inf where p
    keeps an id that q's filter dropped ("support mismatch"). The unfiltered KL over all
    `vocab_used` ids at the same T is reported beside it.
- **Near-tie.** An argmax mismatch counts as a near-tie when the int8 argmax is in the
  top-1 set of the bf16-rounded reference: the golden gate's tie rule
  (`tests/golden/golden_common.h`).

## 2. The table

| | golden | toolcall | all |
|---|---:|---:|---:|
| positions | 99 | 1041 | 1140 |
| logits cosine min | 0.9999148 | 0.9999172 | **0.9999148** |
| logits cosine mean | 0.9999679 | 0.9999719 | 0.9999715 |
| argmax mismatches (near-tie) | 0 | 1 (1) | **1 (1)** |
| top-20 set equal | 90 / 99 = 90.9 % | 959 / 1041 = 92.1 % | **1049 / 1140 = 92.0 %** |
| control: bf16-rounded output, top-20 equal | 74.7 % | 87.1 % | 86.1 % |
| **T 1.0 / k 20 / p 0.95** | | | |
| KL mean | inf (6 rows) | inf (1 row) | **inf (7 rows)** |
| KL mean, finite rows | 8.1e-5 | 4.6e-6 | 1.1e-5 |
| KL p99 | inf (6 > 1 %) | 1.4e-4 | **3.3e-4** |
| support mismatches (control) | 6 (7) | 1 (1) | 7 (8) |
| p's mass on the dropped id, max | 0.020 | 0.011 | 0.020 |
| unfiltered KL mean / p99 / max | 9.8e-5 / 5.3e-4 / 5.7e-4 | 5.3e-6 / 1.4e-4 / 2.4e-4 | 1.3e-5 / 2.0e-4 / 5.7e-4 |
| **T 0.6 / k 20 / p 0.95** | | | |
| KL mean | inf (8 rows) | inf (1 row) | inf (9 rows) |
| KL mean, finite rows | 1.5e-4 | 3.7e-5 | 4.7e-5 |
| KL p99 | inf (8 > 1 %) | 2.7e-4 | 1.1e-3 |
| support mismatches (control) | 8 (8) | 1 (1) | 9 (9) |
| p's mass on the dropped id, max | 0.033 | 0.051 | 0.051 |
| unfiltered KL mean / p99 / max | 1.7e-4 / 1.1e-3 / 1.4e-3 | 8.2e-6 / 2.5e-4 / 7.6e-4 | 2.2e-5 / 4.9e-4 / 1.4e-3 |

**Where the top-20 sets differ.** All 91 mismatched rows differ by exactly **one** id. In
84 of them the reference's rank-20 id is displaced, and in 7 its rank-19 id. On those
rows the bf16 head's logit gap between rank 20 and rank 21 is 0.0099 at the median and
0.046 at most; on all rows the median is 0.069. These are boundary swaps between
near-equal logits. None of them touches the top of the distribution.

**The one argmax mismatch** (`toolcall/t6_usage-openai`) is a near-tie under the golden
rule: the two candidates are equal in bf16, with an fp32 gap of 0.0199.

**Support mismatches.** On each such row, the ids p keeps and q drops hold 0.9-2.0 % of
p at T 1.0, and up to 5.1 % at T 0.6. Near-equal logits moved the top-p cut; no
confident token was lost. The control has the same count, or one more.

## 3. Host quantisation time

The full `[248320, 5120]` tensor quantises to 1,271,398,400 int8 bytes plus 993,280
bytes of scales. That is 1.272 GB, against 2.543 GB for bf16. No row has scale 0. The
round-trip bound `|w - q*s| <= s * (0.5 + 127 * 2^-23)` holds on every element.

| threads | time |
|---|---:|
| 22 (torch in the reference image caps a requested 40 at 22) | **3.27 s** (3.29 s in the first run) |
| 1 | **11.59 s** (10.96 s in the first run) |

That is torch on the box's CPU (Xeon, T5810), reading from bf16 in RAM, with no I/O.
**Derived:** a multithreaded host path in the loader fits under spec 9 §3's 10 s
threshold, so no quantisation kernel on the card is needed. A single-threaded loop sits
right at the threshold.

## 4. Verdict against spec 9 §4

| gate | bar | measured | |
|---|---|---|---|
| L1 cosine | >= 0.9999 | min 0.9999148 | **pass** (1.5e-5 of margin) |
| L1 argmax | equal except bf16 near-ties | 1 mismatch in 1140, a near-tie | **pass** |
| L1 top-20 set | equal in >= 99 % of positions | 92.0 % | **FAIL** by 7.0 points (91 rows against the 11 allowed) |
| L2 KL mean (T 1.0) | <= 1e-3 | +inf (7 support-mismatch rows); 1.1e-5 on the other 1133 | **FAIL** as defined |
| L2 KL p99 (T 1.0) | <= 1e-2 | 3.3e-4 | **pass** |

Under the plan's stopping rule this is a **fail: record it, and keep bf16.**

## 5. What the operator is asked to rule

The two failures are the sampling-boundary bars. On the same rows, a bf16 head whose
output is rounded to bf16, which is the oracle's own logit format, scores **86.1 %**
top-20 equality and **8** support mismatches at T 1.0 (int8: 92.0 % and 7). So the bf16
head itself does not meet the bars as written once only its output rounding changes.
Any second implementation of the bf16 head, with a different accumulation order, would
be graded on the same boundary jitter. Two readings are possible:

1. **The bars stand.** Then int8 is rejected and the head stays bf16. Spec 9b is not
   started.
2. **The bars were meant to measure more than boundary jitter.** A replacement
   candidate, **not applied here**: top-20 equality excluding swaps where the
   reference's rank-20/21 gap is below the bf16 tie tolerance, or below 0.05 logits.
   int8 then passes all 1140 rows (derived from the table above: the maximum gap on a
   mismatched row is 0.046). The KL would be taken on the unfiltered distributions, or
   with the dropped id's mass in place of the inf. int8's unfiltered KL is mean 1.3e-5
   and p99 2.0e-4 (T 1.0), and its dropped mass is at most 0.020. Under that reading,
   int8 passes, and L3 (golden and A4 end to end, in 9b) becomes the real test.

## 6. Reproduce

On the box, from the repo root (`~/b70-inference-server-spec9a`), with >= 70 GB free:

```bash
tools/probe/detach.sh ~/spec9a.log tools/oracle/lm_head_probe.sh               # dump + analyze
SKIP_DUMP=1 tools/probe/detach.sh ~/spec9a-analyze.log tools/oracle/lm_head_probe.sh  # analyze only
tools/oracle/run_in_container.sh 'python3 tools/oracle/test_lm_head_probe.py'  # unit checks
```

Outputs go to `oracle-out-spec9a/`: `hiddens.safetensors` (24 sources, 1140 rows,
bf16 `[rows, 5120]`), `hiddens.safetensors.check.json` (the Review Focus 1 cosine per
source) and `result.json`.

- **Dump:** 22 torch threads, 15633 s wall. The per-source forward runs 30-37 s on a
  golden prompt, 459-702 s on a 960-1144-id scenario, and 1768-2034 s on a ~2200-id one.
  Load and dequantisation take 134 s.
- **Analyze:** a few minutes, and it needs the head only.
