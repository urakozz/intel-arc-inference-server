# Pre-registration - L1-engine (`Engine::prefill`, the CLI `--pp` row, the three gates)

Written and committed **before** any of the numbers below were taken, for the
same reason `docs/prefill-l1-preregistration-2026-09-05.md` was: a prediction
that arrives after its measurement is not a prediction. Every figure here is
labelled **derived** (arithmetic over previously measured rows) or
**estimated** (a guess with its basis stated). Nothing here is measured.

Stage: plan 6b Tasks 10-14, brief
`.superpowers/sdd/2026-09-04-plan6-spec2-prefill/stage1-opus-L1-engine-brief.md`.
Box conditions for everything that follows: `ZE_AFFINITY_MASK=1`, two desktop
processes (baobab, ptyxis) hold DRM fds on every card, so **every timed row is
iterate grade** (8 replays, drop 3, median) and is labelled so. No record row is
claimed anywhere in this stage.

---

## 1. The launch arithmetic - pinned before the walk is run

Derived from the walk in `src/runtime/prefill/step.cc`. `Context::launches()`
counts **L0** launches only; the sycl-tla GEMMs are on the SYCL queue and are
counted separately.

| per | L0 launches | SYCL GEMMs | host `wait()`s |
|---|---:|---:|---:|
| GDN layer (48) | 20 | 4 | 8 |
| FA layer (16) | 15 | 12 | 17 |
| chunk boundary | 1 (embed) | 0 | 0 |
| **one chunk** | **1201** | **384** | **656** |
| `step_head`, once per `prefill()` | 5 | 0 | 0 |

GDN layer: 2 norm + 1 dequant(qkv‖z) + 1 a‖b + 10 `gdn_chunk` + 1
dequant(out_proj) + 2 norm + 1 dequant(gate‖up) + 1 silu + 1 dequant(down).
FA layer: 2 norm + 1 dequant(qkv) + 1 `pf_attn_prep_q16` + 4 `pf_softmax_causal`
(one per kv group) + 1 `pf_attn_gate` + 1 dequant(o_proj) + 2 norm +
1 dequant(gate‖up) + 1 silu + 1 dequant(down).

**None of the three columns depends on `C`.** That is the whole return on the
runtime-`M` rule and on ruling A14 retiring the `C ≤ 64` attention sub-chunk:
plan 6b's own table had the FA family growing as `3·⌈C/64⌉`, i.e. 4182 launches
at C = 4096. This walk is 1201 at every chunk width.

`prefill_smoke_test` asserts `1201·⌈L/C⌉ + 5`.

## 2. Composed attention (`attn_chunk`) - the correctness bars

`attn_chunk` here is **L1 functional, untuned** (the brief's label, carried on
every row that times it). Plan 6d owns the tuned version.

The reference is `tests/kernels/attn_ref.h` - decode's own chain - run two ways,
because A9's bf16 `q` is a **named, expected** numerics change and must not be
allowed to hide a composition bug:

* **Reference A** - `attn_ref` driven with the fp32 `attn_q` decode uses.
  Diagnostic only. It measures A9's cost and nothing else is inferred from it.
* **Reference B** - the same reference driven with `q` pre-rounded to bf16 and
  widened, i.e. the identical input the device GEMM sees. **This is the gate.**
  What remains between B and the device is: DPAS score association vs the
  16-lane tree, whole-row two-pass softmax vs the blocked online one, bf16
  attention weights, DPAS PV association, and normalise-before-PV vs
  divide-after.

Pre-registered, at C ∈ {1, 64, 256} × depth ∈ {0, 4096}, on the gated output
`out` bf16 [C][6144] (post output-gate, pre o_proj):

| | reference B (gate) | reference A (diagnostic) |
|---|---|---|
| max rel (rms-floored) | **≤ 1.0e-2** | recorded |
| relative L2 | **≤ 2.0e-3** | recorded |
| max bf16 ulp | recorded against **A19's 3** | recorded |

Basis for B's bars (derived): the output is a convex combination `Σ pᵢ vᵢ` with
`Σ p = 1`; rounding each weight to bf16 is a ≤ 2⁻⁹ ≈ 2.0e-3 relative
perturbation per weight, and the errors are independent across the row, so the
L2 of the combination moves by ~2.0e-3/√n_eff - under 1e-3 at every depth here.
The 1.0e-2 max-rel headroom is for the smallest output words, where the
rms-floored metric's denominator is the floor and a single bf16 weight's last
bit dominates. **A19's 3-ulp bar is quoted and scored but is NOT the gate**: it
was pre-registered by plan 6d-composed for four association-only changes, and
bf16 attention weights are a value change, not an association.

Two bars that ARE hard, because nothing in them is allowed to differ:
* `kv_k` / `kv_v` written by `pf_attn_prep_q16` **bit-identical** to
  `attn_ref::prep`'s;
* `pf_q` **bit-identical** to `rne_bf16(attn_ref::prep`'s fp32 `attn_q)` - i.e.
  A9 is exactly one rounding at the store and nothing else moved.

## 3. The three gates (spec §6.1, §6.3, §6.4)

* **`prefill_gate_test`** (golden, tie-aware, BOTH checkpoints). Predicted
  **green**, and the basis is not optimism: docs/14 records a 21% hidden-state
  deviation that still produced identical greedy tokens, and the controller's
  L1-core ruling made the token gate the arbiter over `gdn_chunk`'s measured
  3.506e-02 state band for exactly this reason. Prompts are 38-61 ids, i.e. one
  chunk, so the multi-chunk carry is not under test here (Task 13's long gate is).
  **If it is red**, the brief's rule binds: diagnose with the consistency test's
  diagnostics, and if it localises to `gdn_chunk`'s band or A9's bf16-q, STOP
  and report the numbers. The pre-priced first move in that case is L1-core's
  Q1/Q2-fp32 lever, and it is not taken speculatively.
* **`prefill_consistency_test`** (prefill vs decode-ingest, 64 generated ids
  identical). **This is the bar most at risk** and it is stated, not hidden: the
  chunked recurrence is algebraically equal to decode's and differently rounded,
  and the composed attention is a different chain from `attn_decode`/`attn_reduce`
  entirely. Predicted: **64/64 identical on all three prompts at chunk = kC**;
  at chunk = 16 the GDN carry is exercised hardest and a divergence is
  **plausible** - if it happens it is a recorded finding with its first
  divergence position and logit cosine, not a widened bar.
  Named expected contributors to the printed state diagnostics, in advance:
  `gdn_state` at `gdn_chunk`'s measured band (max rel 3.506e-02, mean
  1.197e-03), and `kv_k` at A9's bf16-q - which affects `kv_k` **not at all**
  (k is bf16 in both paths) but does move the logits through `q`.
* **`prefill_determinism_test`** (twice from reset → bitwise). Predicted
  **green with no qualification**: no kernel on this path uses an fp atomic, the
  GEMM is instantiated without split-K, and `gemm_batched_test` already measured
  bitwise repeatability of the batched GEMM. A red here would be a real defect,
  not a rounding question.

## 4. The first `--pp 4096` row - predicted before it is taken

**Composition, per chunk at C = 2048, from the ledger's measured terms**
(progress.md "THE TWO LEVERS, MEASURED" and "THE LAST TWO LEVERS"):
GEMM 680.1 + dequant 210.1 + small kernels 130.3 + GDN 15.4 + lm_head/interop
≈ 8, i.e. **1043.9 ms/chunk before attention** (derived, each term measured
standalone).

**This stage's attention, derived from the measured GEMM cells** (QKᵀ L=6:
51.25 TFLOP/s; PV L=6 strideB=0: 83.68 TFLOP/s; 590 GB/s device bandwidth):

| term (per FA layer, C = 2048) | depth 2048 | depth 4096 |
|---|---:|---:|
| QKᵀ, 4 kv groups | 1.01 ms | 2.01 ms |
| PV, 4 kv groups | 0.62 | 1.23 |
| `pf_softmax_causal` (3 passes over S, 1 bf16 write) | 1.37 | 2.73 |
| `pf_attn_prep_q16` + `pf_attn_gate` | 0.46 | 0.46 |
| **per layer** | **3.46** | **6.43** |
| **× 16 FA layers** | **55.4** | **102.9** |

Plan 6d's pre-registration for its tuned attention is **83.4-88.9 ms/chunk at
depth 4096**. This untuned one is predicted at **102.9 ms**, i.e.
**+14 to +19.5 ms/chunk (+16-23%) worse**, and the mechanism is named in
advance: three passes over `S` instead of a fused one (the softmax term is
2.73 ms/layer where a fused pass would be ~1.4), and no depth-adaptive head tile.

**The term the ledger never priced: cross-runtime handoffs.** 656 host
`wait()`s per chunk at A24's measured 22.35 µs = **14.7 ms/chunk (derived)**.
A24 measured that cost at the L0→SYCL boundary and killed the slab lever with
it; this walk pays it 656 times because the B70 has one compute queue and no
device-side cross-runtime dependency.

**Predicted `--pp 4096`, C = 2048 (two chunks, depths 2048 and 4096):**

| | ms |
|---|---:|
| chunk 1 (pos 0, depth 2048) | 1043.9 + 55.4 + 14.7 = **1114.0** |
| chunk 2 (pos 2048, depth 4096) | 1043.9 + 102.9 + 14.7 = **1161.5** |
| **total (derived)** | **2275.5 ms → 1800 t/s** |

Against the ledger's C = 2048 ceiling of **1808.5-1817.3 t/s**, that is
**−0.5% to −1.0%** - and the near-coincidence is arithmetic, not luck: the
ledger's rate is a per-chunk figure at depth 4096, while a real 4096-token
prefill runs its first chunk at half that depth, and the attention saved there
almost exactly pays for this stage's untuned softmax and the handoffs.

**The honest band, because a composition is not a sum of standalone batteries.**
Every term above was measured in its own probe, warm, with nothing else
resident; in situ each will read worse. Pre-registered range:
**2.28-2.75 s for 4096 ids → 1490-1800 t/s (derived), point estimate
2.40 s / 1707 t/s**, iterate grade. Scoring rule: the row is reported against
this band and against the 1808-1817 ceiling; a MISS is diagnosed by term (the
CLI prints the launch and wait counts, which is the first split), not tuned away.

**vs the 121 s baseline** (bench log `2a7df0b`, RTN, 4096 ids by decode replay):
the predicted 2.40 s is a **50×** cut; the ceiling's 2.26 s is 53×.

## 5. The ≥ 2048-id oracle (Task 13) - cost, estimated

Per plan 6b Task 13 Step 4, restated so the run has something to be scored
against. load + dequant **118 s (measured**, `tools/oracle/README.md`'s table**)**;
prefill forward **~900 s (estimated**, the measured 14.8 s at 42 ids scaled
linearly in T with +10% for the quadratic attention term**)**; 32 greedy
**~250 s (estimated)**; total **20-40 min for one prompt (estimated)**, plus
~18 s of container start (measured). Peak RSS **~70 GiB (estimated)** of the
box's 121 GB; output **~7.3 GB (estimated)**. The run's own `/usr/bin/time -v`
is the measurement, and the estimate stays beside it whichever way it lands.

The multi-chunk gate at C = 1024 over that prompt is registered **whatever it
does**. Plan 6b marked it "expected-limited under L1" because of the retired
`C ≤ 64` attention; that limitation is gone (ruling A14), so the prediction here
is different from the plan's: **green**, with the risk being depth (2400+
positions of chunked GDN carry against 141) rather than attention.
