# W4A8 on `i4_i8` DPAS - probe design

**Status:** design, 2026-09-22. Probe first. This document does not propose
adopting W4A8; it decides whether adopting it is worth designing.

## 1. Why this exists

`docs/probe-dpas-rates-2026-09-22.md` measured this silicon's DPAS rates:
bf16 **183.45 TFLOP/s** (99.97% of the clock-derived peak), `i8_i8` **2.000×**,
`i4_i8` **2.000×**, `i4_i4` **4.000×**, with FP4 refused by the backend and every
`scaled_matrix_mad` form crashing IGC. Integer is the only reachable
lower-precision path on a B70.

The linear path is **1618.8 ms** of a 2493.6 ms GPU total - `slab_gemm` 1344.3 +
`slab_dequant` 274.5 (`docs/prefill-parity-2026-09-20.md`). Our GEMM already runs
at 88.1% of the bf16 peak, and the fused-dequant probe
(`docs/probe-fused-dequant-2026-09-22.md`) showed the memory side is not
recoverable. **bf16 is finished.** `i4_i8` is the only remaining lever on 65% of
the engine's GPU time.

Two properties make W4A8 unusually cheap to try here:

- **No new checkpoint.** The weights stay exactly as they are - int4, symmetric,
  group 64 - and feed the DPAS natively as `:s4`. Only activations are
  quantised, dynamically, at runtime.
- **The dequant pass disappears**, because nothing is widened to bf16 any more.

## 2. What must be measured, and why this probe is not like the last one

The fused-dequant probe could be gated by `memcmp`: it changed no arithmetic.
**W4A8 changes the arithmetic**, so this probe must produce **two** results, and
a win on either one alone decides nothing:

1. **Rate** - is it actually faster once the group-wise rescale is paid?
2. **Error** - how far does the output move from the bf16 path on *real* weights
   and *real* activations?

Neither is a model-accuracy verdict. This probe decides whether to fund an
accuracy study, not whether the model is still correct.

## 3. The arithmetic to implement

Per output tile, for each group `g` of 64 along K:

```
acc_i32 = 0
for the two k-halves of the group:          # K = 32 per i4_i8 instruction
    acc_i32 = intel_sub_group_i4_i8_matrix_mad_k32(w_s4, x_s8, acc_i32)
out_f32 += (float)acc_i32 * (wscale[g][n] * xscale[m])
```

- **Weights**: the existing int4 symmetric g64 values, sign-extended nibbles, fed
  as the 4-bit operand. No dequant, no bf16.
- **Activations**: dynamic **per-token** symmetric int8.
  `xscale[m] = max|x[m][:]| / 127`, quantised RNE, computed by a small kernel
  over the chunk immediately before the GEMM. Report its cost separately - it is
  part of the price.
- **Accumulate** in int32 within a group, rescale into fp32 **at every group
  boundary**. With g64 and K=32 that is exactly two instructions per rescale.

**The cost this probe exists to find.** At K = 5120 there are 80 groups, so each
output element pays 80 int32→float conversions and 80 fp32 FMAs on top of the
matrix math. That ALU is the direct analogue of what sank the fused-dequant
probe, and it is why a 2× instruction rate does not imply a 2× kernel. Measure it
before believing it.

## 4. Break-even, stated before measuring

The same-shape control is already measured
(`docs/probe-fused-dequant-2026-09-22.md`): gate‖up, K = 5120, N = 34816,
M = 2048 - dequant 0.826 ms + GEMM 4.515 ms = **5.341 ms**, 136.72 TFLOP/s
effective, 730.1 GFLOP.

| outcome | W4A8 time for the shape | implication |
|---|---:|---|
| break-even | 5.341 ms | no reason to continue |
| **worth a study** | **≤ 3.5 ms** (≥ 1.53×) | linear path ≈ 1060 ms, −560 ms |
| the instruction's promise | 2.67 ms (2.00×) | linear path ≈ 810 ms, −810 ms |

**The probe fails below 1.53×** and that is recorded rather than tuned away.

## 5. Real inputs, not synthetic

Activation outliers are the entire risk in W4A8, and random tensors do not have
them. The probe must use:

- **Real weights**: gate‖up of one real layer, through the loader.
- **Real activations**: one chunk of `s.x` captured from an actual prefill of a
  golden prompt at C = 2048, dumped once and reused, so the error numbers are
  about this model rather than about a distribution someone invented.

Report, against the bf16 two-pass result on the same inputs:

- max and mean relative error per output;
- per-row cosine, and the **worst row**;
- the fraction of outputs where the bf16 and W4A8 argmax-relevant magnitude
  ordering changes, if cheap to compute;
- the observed activation dynamic range per token - max|x| over median|x| - since
  that number, not the error, predicts how this behaves on other layers.

## 6. Evidence required

- Rate for: the bf16 control, W4A8, and the activation-quantise kernel alone.
- The error table of §5.
- Assembly for the W4A8 mainloop: the `dpas` count per group, that operands are
  native `:s4`/`:b` with no widening, GRF allocation, spill bytes, and **the
  rescale sequence** - how many instructions per group boundary, since that is
  the measured cost of §3's warning.
- Idle-box statement and device (device 1; device 0 is the series card).

## 7. Decision rule

- **< 1.53×**, or the rescale dominates → **rejected**, recorded, W4A16 stands
  and the parity program's S2-S5 remain the whole plan.
- **≥ 1.53× with small error** → write the adoption spec, whose first section
  must be the *proving strategy*, not the kernel: a W4A8 CPU oracle, task-level
  acceptance (the current 93/93-against-bf16 contract cannot survive an
  arithmetic change), and what happens to `prefill_gate`, `prefill_consistency`,
  `prefill_determinism` and `prefill_backend_equivalence_test`.
- **≥ 1.53× with large error** → report both numbers and stop. A fast kernel that
  needs a weaker gate is not an optimisation, it is a different model.

## 8. Out of scope

`i4_i4` (W4A4) - it needs int4 activations, and the accuracy question is an order
harder; this probe's error numbers are the evidence that decides whether it is
ever worth asking. GDN, attention, decode, the KV cache, the checkpoint format,
and every model gate. No production file is edited: new probe `.cc` + `.cl` and
appended CMake lines only.
