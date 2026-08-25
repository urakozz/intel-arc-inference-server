# 14. The golden gate - the trust chain closed

Every test before this one compares a kernel against a host reference **this
project wrote**. `gemv_test` grades `gemv.cl` against `gemv_ref.h`;
`gdn_step_test` grades `gdn_step.cl` against `gdn_ref.h`. Those tests find
transcription bugs, and they cannot find a shared misunderstanding: a reference
and a kernel written from the same wrong reading of the modeling file agree
perfectly with each other and disagree with the model.

The golden gate is the test that closes that hole. It compares the **whole
engine** - 645 kernels, the loader, the int4 repack, the captured command list,
the decode loop - against a reference the project did not write: `transformers`
5.15 on CPU, the pure-torch gated-delta-rule, `attn_implementation="eager"`,
over the same checkpoint dequantised by the same convention the C++ loader is
bit-compared against (`tools/oracle/README.md`).

**Result, measured 2026-08-25 on the box** (`box`, Arc
Pro B70, `tests/golden/golden_gate_test.cc`):

> **3 prompts × 32 greedy tokens, 96/96 token ids element-exact, first run, no
> engine change required.**

Nothing in the engine was modified to make the gate pass. `.cl` edits were
budgeted for this task and none were spent.

## What the gate actually asserts

| | Bar | Measured |
|---|---|---|
| 32 generated ids == golden `tokens`, per prompt | **exact - this is the gate, and the only thing that fails the test** | 32/32, 32/32, 32/32 |
| `gdn_state` after the prompt, per GDN layer, cosine | diagnostic, `**LOW**` below 0.999 | min **0.999537**, 0/144 layer-prompt pairs below the bar |
| per-layer residual tap cosine | diagnostic, `**LOW**` below 0.999 | see below |
| logits row at every decision point, cosine | diagnostic | 0.999844 … 0.999989 over all 96 rows |

**Token equality is the gate. Every cosine in this document is a diagnostic** -
they mark rows `**LOW**` and they do not fail the run. That is spec §11's
ruling: "64 layers of bf16 residual drift make a hard tensor bound either
useless or flaky". §11's own wording, written before any of this was measured,
is what the `code` prompt below turned out to demonstrate.

`gdn_state` was written as a hard bound (the Task 8 brief states a ≥ 0.999 bar
for it) and the Task 8 review **demoted it to a diagnostic**, 2026-08-25, on
§11's authority. The measured margin at the moment of that ruling was 5.4e-4 -
worst 0.999537065, `code`, L60 - so the bound would have been one legitimate
numerics change, or one longer prompt, away from failing the golden gate on a
recurrent-state cosine while the token ids were exact. It still prints, still
marks `**LOW**` per layer, and `tests/golden/golden_gate_test.cc`'s `kBar`
comment carries that margin and the reason.

### Per prompt

| Prompt | ids | tokens exact | tap min cos (layer, t) | tap median cos | L63 tail min cos | `gdn_state` min cos | logits min cos |
|---|---|---|---|---|---|---|---|
| prose | 42 | **32 / 32** | 0.999736426 (L34, t=14) | 0.999965335 | 0.999884841 | 0.999910420 (L49) | 0.999920374 |
| code | 61 | **32 / 32** | 0.986308437 (L59, t=41) | 0.999967293 | 0.996597482 | 0.999537065 (L60) | 0.999843703 |
| cjk | 38 | **32 / 32** | 0.999775022 (L51, t=0) | 0.999969090 | 0.999865750 | 0.999911630 (L33) | 0.999893648 |

All measured. "tap median cos" is the mean over the 64 layers of each layer's
**upper-median** over positions (element `T/2` of the sorted row; for even `T`
that is the upper of the two central values, not their mean) - the typical
comparison, next to the worst one. The test prints it, so the figure is read
off a log rather than derived by hand.
The engine's continuations are the three recorded in `tools/oracle/README.md`
("Sanity checks on the written files"), id for id, including the CJK prompt's
completion of a trailing emoji variation selector.

## How the tap is compared - and why it is not a layer output

The engine's per-layer tap is **not** layer *i*'s output. `runtime/capture.h` is
the single authority: the copy runs after layer *i*'s last kernel, and the
residual stream is advanced only by `prep_res_norm`, so `tap[i]` holds the
hidden state with layer *i*'s **mixer** folded in and layer *i*'s MLP still
sitting un-folded in `partials` - layer *i+1*'s leading `prep_res_norm` folds
it. The oracle's `resid.L{i}` is the layer *output*, post-MLP. Comparing the two
directly would report a divergence that is a definition mismatch and nothing
else. The comparator is therefore **built** from two golden tensors rather than
read from one:

```
expected_tap[i][t] = bf16( resid.L{i-1}[t] + mixer.L{i}[t] )     i >= 1
expected_tap[0][t] = bf16( embed[ids[t]]   + mixer.L0[t]   )
```

summed in fp32 from the two golden bf16 tensors and rounded RNE
(`common/bf16.h`), because that is the arithmetic `prep_res_norm` does. The
oracle dumps no embedding tensor, so layer 0's `resid.L-1` is gathered
host-side out of `LoadedModel::embed` - the same bf16 table the device gathers
from, so that row contributes no error of its own.

**Layer 63's post-MLP residual has no tap** (there is no tap 64) and none is
invented. It is covered twice over instead: `b.resid` after the fence *is* that
vector (capture.h), so the test compares it against golden `resid.L63`
directly - the "L63 tail" column above - and the logits and the token ids cover
it again.

## The named divergence classes, and what they measured

Two divergences were named in advance as expected. Both were measured here.

### 1. GEMV accumulation order and the dequantisation rounding point

The engine never materialises a weight: `gemv.cl` computes `scale · Σ(q − 8)·x`
with the products and the sum in fp32 and the scale applied once per group of
64. The oracle materialises `bf16(scale · (q − 8))` and lets torch multiply bf16
weights by activations. Two different accumulation orders, two different
rounding points, on identical bits - the engine is the *more* accurate of the
two, and `tools/oracle/README.md`'s trust chain predicts roughly **0.2% RMS per
weight**. That 0.2% is **estimated, not measured**: it is the analytic cost of
the oracle's extra per-weight RNE cast to bf16 - an 8-bit mantissa gives a
relative rounding error of about 2⁻⁹ ≈ 0.2% RMS, which the engine does not pay
because it never materialises a weight. No plan-2 measurement of a per-weight
RMS exists to cite. What *is* measured is the effect of that class on an actual
activation, below.

The gate measures the seed directly. The embedding gather is an exact bf16 row
copy, so **every** bit of the layer-0 tap error is layer 0's mixer - its GEMVs
and its GDN step:

| Prompt | layer-0 tap: max relative L2 error | min cos |
|---|---|---|
| prose | 4.468e-03 | 0.999999005 |
| code | 5.429e-03 | 0.999998601 |
| cjk | 6.837e-03 | 0.999994722 |

**0.45 - 0.68% max relative on one mixer output** - all three measured, and the
same order as the estimated 0.2% RMS per weight above. An accumulation of ~5000
weight-level errors of that size into one dot product, then through a norm and a
second GEMV, landing at half a percent, is what "expected class" means here.

### 2. The softmax path

The oracle runs `attn_implementation="eager"`: an fp32 softmax over the whole
row. The engine runs a blocked flash-decode with per-block running max/sum
merged by `attn_reduce`. The `exp` implementations differ, and the merge order
differs from a single-pass softmax. This class was expected to show up as FA
layers being systematically worse than GDN layers. **It does not:**

| Prompt | GDN layers, mean of per-layer medians | FA layers, mean of per-layer medians |
|---|---|---|
| prose | 0.999965827 (48 layers) | 0.999963861 (16 layers) |
| code | 0.999967797 | 0.999965782 |
| cjk | 0.999969695 | 0.999967274 |

The gap is 2e-6 - two parts per million, three orders of magnitude below the
0.999 bar, and in the same direction on all three prompts only because the FA
layers sit deeper in the stack on average. All measured. **The softmax path
contributes no measurable divergence of its own** at this depth; class 1 is the
whole story.

### A third thing that could have diverged, and did not

`lm_head` is `[5120][248320]` for tiling, but only 248077 of those columns are
real tokenizer ids; `argmax.cl` masks the 243-column tail to `-INFINITY`. The
oracle argmaxes the full 248320. If a padding column had ever won a row, the two
would have disagreed by construction. The test reports the golden argmax both
masked and unmasked at every decision row: **they were equal in all 96 rows**,
so the mask is not a divergence source on these prompts. Measured.

## The `code` prompt's LOW rows - the interesting result

The `code` prompt is the only one with tap cosines below 0.999, and the shape of
the failure is worth the space, because it is exactly what spec §11 anticipated.

**Census: 36 of the 3904 (layer, position) tap comparisons are below 0.999, on 3
of the 61 positions - t = 17, 41 and 43.** prose is 0 of 2688 and cjk is 0 of
2432. Every layer's *median* over positions is ≈ 0.99994, layer 63 included; the
minima and the medians differ by three orders of magnitude in the error.

The worst is L59 at t=41, cosine 0.986308. Two facts kill the two cheap
explanations:

- **It is not a small denominator.** `|oracle|` at (L59, t=41) is 275.7, dead
  centre of that layer's distribution over positions (189 … 408). The error
  `|engine − oracle|` there is **58.8** against a typical **≈ 3**: a 20×
  outlier in absolute terms, not a normalisation artefact.
- **It is not one layer, and it is not one kernel.** The per-layer trace at
  t=41 grows monotonically and smoothly with depth - |err| 0.007 (L0), 0.407
  (L20), 1.017 (L40), 2.026 (L45), 4.875 (L50), 33.1 (L55), 58.8 (L59) - and
  then *recovers* in relative terms (L63: |err| 32.3 on |oracle| 397, cos
  0.9967). There is no step change at any single layer and none at any layer
  *kind*. A kernel defect would put the step somewhere.

What it is: **one token position whose residual direction the network amplifies
strongly**, applied to the class-1 seed that every position carries. The test
prints the attribution that shows it - for each layer, alongside the tap error,
the cosine of the layer's own contribution (`tap[i] − tap[i−1]` on the engine
side, `mixer.L{i} + mlp.L{i−1}` on the oracle side, both available in the golden
file). At t=41 the contribution cosine sags to 0.98211452 at L55 and 0.89561621 at
L60 while the accumulated tap is still at 0.99, and L60's contribution -
`|contrib|` 107.122 with `|contrib err|` 48.689 - *pulls the tap back up* from
0.9863 to 0.9936. A layer
whose output is a strong function of the residual direction, with both engine
and oracle falling into the same attractor. That is high local gain, not a bug.

Three independent facts say the amplification is harmless where it matters:
`code`'s `gdn_state` cosines are all ≥ 0.999537 (the perturbation does not
poison the recurrent state), positions t=42…60 are all back at ≈ 0.9999 (it does
not propagate along the sequence), and the 32 generated ids are exact.

This is why the gate is token equality. A hard 0.999 tensor bound would have
failed this run - on a prompt where the engine reproduces the oracle's output
perfectly.

## Determinism

The gate was run nine times on 2026-08-25 - six standalone and three under
`ctest` - across five builds of the test binary (three adding diagnostics, one
for the review fix round). **Every run produced the same 96 token ids**, and
every quantity two kept logs have in common agrees to **all nine printed
digits**; the diagnostics only ever grew between builds, they never moved. That is the replayed
command list's bitwise reproducibility (Task 6's `replay_determinism_test`)
observed end to end, on the real checkpoint, through a 61-token ingest and 32
generated tokens - measured.

## The CLI cross-check - closed 2026-08-25

The gate proves the **engine**. It does not run `b70-decode`: it links
`runtime::Engine` directly, reads the `.ids` files itself and compares in
process. So until this check, the shipped binary - its argument parsing, its
`--ids` reader, its ingest/generate call order, its stdout channel - had never
been graded against anything.

```bash
tools/box.sh run "./build/src/cli/b70-decode Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ \
  --ids tests/golden/prompts/prose.ids --n 32 > /tmp/cli_prose.out"
```

stdout, all 32 ids, compared against the `tokens` tensor read straight out of
`oracle-out/prose.golden.safetensors` (dtype I32, shape [32]) rather than
against a derived file:

```
3113 7810 279 1118 479 654 8980 1000 381 1142 440 279 1834 725 2213 13
3113 11292 279 4220 6092 1000 381 6992 11 321 539 5600 279 72103 1000 381
```

**32/32 equal.** The CLI reproduces the oracle's continuation exactly, on the
same prompt, through the product binary. Its stderr on that run -
`ingest: 42 ids in 1600.7 ms (38.11 ms/token)`, `generate: 32 ids, 25.97 t/s`
- is also the first end-to-end confirmation that the `--ids` path and the
`--bench` path measure the same engine.

This is a **check, not a test**: it is a documented command with a recorded
result, and no new test binary was built for it. The gate remains the
regression barrier.

## The vLLM cross-check - status: still open, now actionable

`tools/oracle/README.md`'s trust chain is explicit that the golden gate has a
ceiling: the C++ loader and the oracle are pinned to the **same** `dequant.py`
convention (bit-exactly, by `dequant_fixture_test`), so a convention that is
wrong about how the checkpoint's author packed the nibbles moves the engine and
the oracle **together** and this gate still passes. Closing that needs a third
implementation that shares nothing with either: vLLM's own greedy output on the
same checkpoint.

That cross-check **has not been run.** It was deferred out of plan 2
deliberately - it needs a working engine to be worth running - and this task
did not run it either. What changed today is that it is now actionable, and the
three-way reading is no longer hypothetical:

- **engine ≠ oracle** → an engine bug. The oracle is the reference. *(Ruled out
  today for these three prompts: 96/96 exact.)*
- **engine == oracle, both ≠ vLLM greedy** → **suspect `dequant.py` first**,
  because the convention is the one thing the engine and the oracle share and
  vLLM does not: zero point (`q − 8` vs an explicit `qzeros`), the group axis,
  the `[K/8, N]` packing and nibble order, `desc_act`/`g_idx`. Do not touch a
  kernel first.
- **all three agree** → the chain is closed.

Running it means a vLLM greedy generation on
`Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ` with the three committed prompt id
files and 32 tokens, temperature 0, and comparing against
`tests/golden/prompts/*.ids` → the ids in `tools/oracle/README.md`. It is a
short job on the box and it is the last link.

## Running the gate

```bash
tools/box.sh test golden_gate_test          # sync + build + ctest on the box
# or, for the full diagnostic log:
tools/box.sh run './build/tests/golden_gate_test "$PWD/oracle-out" "$PWD/tests/golden/prompts"'
```

The prompt ids are committed (`tests/golden/prompts/{prose,code,cjk}.ids`, 42 /
61 / 38 ids); the ~1 GB of golden `.safetensors` are not, and live only at
`~/b70-inference-server/oracle-out/` on the box. Absent goldens make the test
exit 77, which ctest reports as **SKIP** - verified by hiding the directory and
re-running. The test also asserts that each `.ids` file has as many ids as its
golden file's `resid.L0` has rows, so a stale ids file cannot silently pass.

**Measured cost** (fix-round run, 2026-08-25, the run every number in this
document comes from): **24.12 s** under ctest, **23.87 s** standalone, of which
**13.4 s** is the checkpoint load; peak RSS **20,489,588 KB = 19.5 GiB**
(`/usr/bin/time -v`). Full suite: **23/23 tests, 105.01 s**. One checkpoint load
serves all three prompts.
