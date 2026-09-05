# `prefill_consistency_test` is red in 5 of 18 cases - the priced record

Spec 2 §6.3's bar is "prefill and decode-ingest generate the same 64 greedy ids,
element-exact". It is red in 5 of 18 (prompt × chunk × checkpoint) cases. Per
the L1-engine brief, a red gate is a finding to diagnose and **not a tolerance
to loosen**, and it has not been loosened: the test still fails, its message
carries the diagnosis, and what follows is the ruling request.

Measured 2026-09-05 on the box, `ZE_AFFINITY_MASK=1`, commit `d93205d`.

---

## 1. What is red

Nine (prompt, chunk) cases per checkpoint: prose/code/cjk × chunk
{2048 = kC, 1024, 16}.

**Vishva** (`B70_TEST_SNAPSHOT`) - 7 of 9 green:

| prompt | chunk | identical/64 | first diff | logit-row cosine | prefill top-2 gap |
|---|---:|---:|---:|---:|---:|
| prose | 2048 / 1024 / 16 | 64/64 | - | 1.000000000 | - |
| code | 2048 | 41/64 | **41** | 0.999976957 | 1.659e-02 |
| code | 1024 | 41/64 | **41** | 0.999976957 | 1.659e-02 |
| code | 16 | 64/64 | - | 1.000000000 | - |
| cjk | 2048 / 1024 | 64/64 | - | 1.000000000 | - |
| cjk | 16 | 13/64 | **13** | 0.999981933 | 1.619e-03 |

**RTN** (`B70_RTN_SNAPSHOT`) - 6 of 9 green:

| prompt | chunk | identical/64 | first diff | logit-row cosine | prefill top-2 gap |
|---|---:|---:|---:|---:|---:|
| prose | 2048 / 1024 / 16 | 15/64 | **15** | 0.99994 / 0.99995 | 3.894e-02 / 3.929e-04 |
| code | 2048 / 1024 / 16 | 64/64 | - | 1.000000000 | - |
| cjk | 2048 / 1024 / 16 | 47/64 | **47** | 0.99998 | 7.159e-02 / 2.396e-02 |

## 2. Every divergence is a sub-ulp coin toss, and here is the number

At each first divergence the test prints both rows' top-1 and top-2 with the
gap, in units of a bf16 ulp of the top value. **Decode's own gap is 0.02-0.32
bf16 ulp in every one of the four distinct rows** - the two candidates are
within a bf16 ulp of each other in decode's fp32 logits:

| case | row | ids | decode top-2 gap | prefill top-2 gap |
|---|---:|---|---:|---:|
| Vishva code | 41 | 379 vs 1030 | 1.812e-03 = **0.03 ulp** | 1.659e-02 = 0.23 ulp |
| Vishva cjk | 13 | 101650 vs 112338 | 4.089e-03 = **0.06 ulp** | 1.619e-03 = 0.02 ulp |
| RTN prose | 15 | 271 vs 353 | 2.502e-02 = **0.32 ulp** | 3.894e-02 = 0.50 ulp |
| RTN cjk | 47 | 104844 vs 111164 | 1.010e-03 = **0.02 ulp** | 7.159e-02 = 1.17 ulp |

A greedy walk resolves such a row by whichever path's last bit lands higher, and
after it the two walks free-run apart - which is the whole of the 41/64, 13/64,
15/64 and 47/64. The identical-prefix counts are not a measure of how wrong the
prefill is; they are a measure of how long ago the coin was tossed.

## 3. The decisive fact: two of the four rows are ones the ORACLE cannot decide either

`golden_decision` (the controller's 2026-08-26 tie ruling, `golden_common.h`)
calls a golden row UNDETERMINED when the oracle's top-1 is not unique - its
logits are bf16 widened to fp32, so two candidates can be the same word. Of the
four divergence rows, two fall inside the oracle's 32-row window, and **both of
them are undetermined rows of that very golden file**:

* `prefill_gate_test` on **RTN prose** prints "undetermined rows in this golden
  file: 3 of 32 - step 15 step 26 step 29". The RTN prose consistency
  divergence is at **step 15**.
* `prefill_gate_test` on **Vishva cjk** prints "undetermined rows in this golden
  file: 2 of 32 - step 13 step 26". The Vishva cjk divergence is at **step 13**.

The other two rows (41 and 47) are past the oracle's 32 generated steps, so the
reference has nothing to say about them either way.

## 4. Three measured facts that come with it

* **`prefill_gate_test` is GREEN on both checkpoints and at every chunk width
  tested.** 94/94 determined rows exact (Vishva) and 93/93 (RTN) at chunk = kC;
  62/62 at chunk 16 and again at 1024 on cjk+code. Where the two walks disagree,
  the prefilled engine matches the CPU oracle. **The oracle is the arbiter;
  decode-ingest is not.**
* **Neither walk is unstable.** `prefill_determinism_test` passes on both
  checkpoints - 9 cases × 3 runs, gdn_state/conv_ring/kv_k/kv_v/control and
  `cur_token` bitwise identical, `PrefillScratch` never re-zeroed - and
  `replay_determinism_test` passes for decode. These are two deterministic walks
  that differ, not one that wanders.
* **The failure is not monotone in chunk width.** Vishva `code` is 64/64 at
  chunk 16 and 41/64 at 2048; Vishva `cjk` is the other way round; RTN `prose`
  and `cjk` are identical at all three widths. A chunking defect gets worse with
  more chunks. A coin toss does not care, and neither does this.

## 5. The named, expected contributors - with their numbers

Both are rulings, recorded in advance of this stage, not discoveries:

* **`gdn_chunk`'s chunked-vs-recurrent algebra** (ruling A22). L1-core measured
  `gdn_state` max rel **3.506e-02**, mean **1.197e-03**, against a CPU fp32
  reference, and attributed the tail to cancellation at the Q1/Q2 bf16 rounding
  points. This test measures the same quantity against **decode on the device**:
  max rel 0.60-1.44, **mean 6.3e-04 to 8.9e-04**. The means agree to within a
  factor of 1.5 of L1-core's; the maxima are larger because the denominator is
  rms-floored over a 37.7 M-word state whose smallest entries are ~0.
* **Ruling A9's bf16 `q`.** It cannot move `kv_k` - k is bf16 on both paths, and
  `attn_chunk_test` holds the cache bit-identical to `attn_ref::prep` - but it
  moves the scores, and therefore the logits, on every FA layer.

`conv_ring`'s max rel of exactly 1.0 in every row is **not** a finding: the
chunked path writes only the three live ring slots (plan 6b's design, and
`gdn_chunk_test` case 5's `live_slots_equal` is the standing proof that nothing
reads the other thirteen), so the comparison includes dead slots that decode
filled and prefill did not. It is left in the table rather than filtered,
because a filtered diagnostic that agrees by construction says nothing.

## 6. The ruling request

The bar as literally written - 64 element-exact greedy ids from two
differently-rounded implementations - **asserts an answer that neither reference
contains** on a row where the top two logits are the same bf16 word. That is
precisely the objection the controller settled for the golden gate on
2026-08-26, when a golden row with a non-unique top-1 became UNDETERMINED and
the walk became teacher-forced after the first divergence.

Three options, priced, **none of them taken here**:

1. **Import the golden gate's two mechanisms.** Teacher-force the prefill walk
   on decode's token after the first divergence (so every later row is graded on
   a shared context rather than on a diverged one), and treat a row whose
   decode-side top-2 gap is under one bf16 ulp as undetermined, requiring only
   that prefill's id be one of the two. Cost: ~20 lines in
   `prefill_consistency_test.cc`. This makes the bar say what it was for.
2. **Keep the bar and record it as expected-limited**, the way plan 6b Task 13
   expected the long gate to be. Cost: nothing, but a permanently red test is a
   bad instrument - it stops being read.
3. **Chase the band to zero.** L1-core's pre-priced first move is the Q1/Q2-fp32
   lever: extend fp32 retention past `gdn_chunk`'s two bf16 rounding points,
   which L1-core measured would remove most of the state band. It trades
   "decode's rounding points" - which A6 mirrors precisely so this
   self-consistency bar means something - for a tighter band. Cost: one task,
   and it would need the whole numerics band re-measured. **It would not close
   this**: A9's bf16 `q` and the composed attention's bf16 weights are
   independent contributors, and a sub-ulp logit row will still be a coin toss.

The controller's call. Nothing in this stage widened the bar, and the test
fails loudly with all of the above in its own output.
