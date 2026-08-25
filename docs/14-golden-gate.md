# 14. The golden gate - the trust chain closed

Every test before this one compares a kernel against a host reference **this
project wrote**. `gemv_test` grades `gemv.cl` against `gemv_ref.h`;
`gdn_step_test` grades `gdn_step.cl` against `gdn_ref.h`. Those tests find
transcription bugs, and they cannot find a shared misunderstanding: a reference
and a kernel written from the same wrong reading of the modeling file agree
perfectly with each other and disagree with the model.

The golden gate is the test that closes that hole. It compares the **whole
engine** - 774 kernels, the loader, the int4 repack, the captured command list,
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

### The diagnostics move with every lever; the gate does not

**The table above is the engine at `adc2544`, and only the `exact/32` column has
stayed put.** Spec 1.5's levers reorder floating-point sums - L2 split the `a‖b`
GEMV's K, L1 split `prep_res_norm`'s Σx² - and each such change moves `x` in the
last bf16 ulp somewhere, which the next 60-odd layers amplify. Two gate runs on
an otherwise idle box, 2026-08-25, the second at the commit that landed L1:

| prompt | | exact/32 | tap min cos (layer, t) | L63 tail | `gdn_state` min cos | logit min cos |
|---|---|---|---|---|---|---|
| prose | before L1 (`b15f70f`) | **32/32** | 0.999448759 (L62, t=24) | 0.999718243 | 0.999905491 (L49) | 0.999924384 |
| prose | after L1 | **32/32** | 0.994806738 (L62, t=24) | 0.997115152 | 0.999851574 (L60) | 0.999864052 |
| code | before L1 (`b15f70f`) | **32/32** | **0.829087356** (L59, t=41) | 0.964064662 | 0.997263854 (L60) | 0.999666022 |
| code | after L1 | **32/32** | **0.913533675** (L59, t=41) | 0.982882165 | 0.998530392 (L60) | 0.999818731 |
| cjk | before L1 (`b15f70f`) | **32/32** | 0.999775022 (L51, t=0) | 0.999830388 | 0.999908549 (L33) | 0.999906932 |
| cjk | after L1 | **32/32** | 0.999690168 (L51, t=0) | 0.999832133 | 0.999904153 (L33) | 0.999876394 |

**Lever L5 (attention `ATTN_BLOCK` 256 → 64, `c746840`) is the first one that
moved nothing.** Two more gate runs on the same idle box, 2026-08-25, before at
`0bfa891` and after at the commit that landed the retile:

| prompt | | exact/32 | tap min cos (layer, t) | L63 tail | `gdn_state` min cos | logit min cos |
|---|---|---|---|---|---|---|
| prose | before L5 (`0bfa891`) | **32/32** | 0.994806738 (L62, t=24) | 0.997115152 | 0.999851574 (L60) | 0.999864052 |
| prose | after L5 | **32/32** | 0.994806738 (L62, t=24) | 0.997115152 | 0.999851574 (L60) | 0.999864052 |
| code | before L5 (`0bfa891`) | **32/32** | **0.913533675** (L59, t=41) | 0.982882165 | 0.998530392 (L60) | 0.999818731 |
| code | after L5 | **32/32** | **0.913533675** (L59, t=41) | 0.982882165 | 0.998530392 (L60) | 0.999818731 |
| cjk | before L5 (`0bfa891`) | **32/32** | 0.999690168 (L51, t=0) | 0.999832133 | 0.999904153 (L33) | 0.999876394 |
| cjk | after L5 | **32/32** | 0.999690168 (L51, t=0) | 0.999832133 | 0.999904153 (L33) | 0.999876394 |

Every column is identical to all nine printed decimals, on all three prompts.
After L2 and L1 each walked `code`'s worst tap across 0.9863 → 0.8291 → 0.9135,
that is a striking result - and **it should be read as a limit of the gate, not
as a property of the lever.**

**Why it did not move, and what that costs.** L5 reassociates the attention
softmax's partials: a work-group now accumulates 4 waves into a partial where it
accumulated 16, and `attn_reduce` merges four times as many of them. But the
gate's prompts are **42, 61 and 38 ids** and each generates 32, so the deepest
context any of them reaches is `pos` **92**. At `ATTN_BLOCK` 64 that is two
blocks; at 256 it was one. The gate therefore exercises a **two-block merge
against a one-block merge** - genuine coverage of the reassociation, and about
3% of the 65-block merge the engine runs at depth 4096. Before the retile
shipped at 64 the same runs were done at 128, where the gate's depths fall
inside a *single* block and the change is provably a no-op there (a wholly
masked wave leaves `(mx, sm, acc)` exactly, `resc = 1` and `fma(acc, 1, 0)`),
so those runs were a control and nothing more.

**What does cover the deep merge** is `tests/kernels/attn_test.cc`, at
`pos = 4095` (64 live blocks) and `pos = 16383` on the L16384 binary (256 live
blocks, a 256-step merge), where the device is held to a ruled **≤ 2 bf16 ulp**
and measures **1 ulp on one word** at its worst, against a host reference that
models the same blocking - and nothing else does. There is
no oracle at depth 4096. **A lever that changes attention arithmetic cannot be
signed off by this gate alone**, and the next one should say so before it starts
rather than after.


Three things this settles, and one it does not.

1. **The gate held, both times, on all three prompts.** 96/96 token ids
   element-exact. That is the whole test; everything else on this page is a
   diagnostic and §11's ruling is what makes that the right arrangement.
2. **The movement has no direction.** L1 made `code`'s worst tap *better*
   (0.829 → 0.914) and `prose`'s *worse* (0.99945 → 0.99481). It is the
   chaotic sensitivity of a handful of ill-conditioned positions -
   `code` t=41, `prose` t=24 - not a degradation. The census bears it out: after
   L1, 44 of `code`'s 3904 (layer, t) comparisons are below 0.999, on **3 of its
   61 positions**, and `cjk` has none at all.
3. **The `b15f70f` column is itself already far from the table above** -
   `code`'s worst tap reads 0.9863 there and 0.8291 at `b15f70f`, and nothing
   between them touched `prep`. That drift is **L2's**, which reported "96/96,
   unchanged" and did not re-measure these cosines. So the diagnostics in the
   table above should be read as *the engine of `adc2544`*, not as a standing
   contract.
4. **What it does not settle**: whether any of this is a *trend*. Four levers
   in, the worst tap cosine has been 0.9863, 0.8291, 0.9135 and 0.9135 (L5 moved
   it by less than 1e-9) without a token ever moving, and there is no model here
   for how far it can go before one does. The honest position is the one §11
   already took - the tokens are the bar - and the practical consequence is that
   **every lever that reorders a sum runs the gate, and records both columns, as
   this one did.** L5 adds a second consequence: a gate run that shows *no*
   movement has to be checked for whether the change was reachable at all from
   42/61/38-id prompts, because L5's was barely.

The engine's continuations are the three recorded in `tools/oracle/README.md`
("Sanity checks on the written files"), id for id, including the CJK prompt's
completion of a trailing emoji variation selector.

## How the tap is compared - and why it is not a layer output

The engine's per-layer tap is **not** layer *i*'s output. `runtime/capture.h` is
the single authority: the copy runs after layer *i*'s last kernel, and the
residual stream is advanced only by `prep_res_fold` (`prep_res_norm` until spec
1.5's lever L1 - the per-element arithmetic is the same one, which is why the
comparator below is unchanged), so `tap[i]` holds the hidden state with layer
*i*'s **mixer** folded in and layer *i*'s MLP still sitting un-folded in
`partials` - layer *i+1*'s leading `prep_res_fold` folds it. The oracle's `resid.L{i}` is the layer *output*, post-MLP. Comparing the two
directly would report a divergence that is a definition mismatch and nothing
else. The comparator is therefore **built** from two golden tensors rather than
read from one:

```
expected_tap[i][t] = bf16( resid.L{i-1}[t] + mixer.L{i}[t] )     i >= 1
expected_tap[0][t] = bf16( embed[ids[t]]   + mixer.L0[t]   )
```

summed in fp32 from the two golden bf16 tensors and rounded RNE
(`common/bf16.h`), because that is the arithmetic `prep_res_fold` does (and
`prep_res_norm` before it, unchanged). The
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

## The vLLM cross-check - closed 2026-08-25

`tools/oracle/README.md`'s trust chain is explicit that the golden gate has a
ceiling: the C++ loader and the oracle are pinned to the **same** `dequant.py`
convention (bit-exactly, by `dequant_fixture_test`), so a convention that is
wrong about how the checkpoint's author packed the nibbles moves the engine and
the oracle **together** and this gate still passes. Closing that needs a third
implementation that shares nothing with either: vLLM's own greedy output on the
same checkpoint.

**It has now been run.** Result, measured 2026-08-25 on the box, image
`vllm-xpu-env-next-p314-t214-vxkp0:latest`, vLLM
`0.27.2rc1.dev365+g5ee84d3c5.d20260821` / torch `2.14.0+xpu` on the Arc Pro B70:

> **3 prompts × 32 greedy tokens, 96/96 token ids element-exact against the
> oracle's golden `tokens` - and therefore against the engine, which the gate
> already proved equal to it, id for id. The trust chain is closed end to end.**

The three readings were named in advance. This is which one landed:

- **engine ≠ oracle** → an engine bug. The oracle is the reference. *(Ruled out
  2026-08-25 for these three prompts: 96/96 exact, top of this document.)*
- **engine == oracle, both ≠ vLLM greedy** → **suspect `dequant.py` first**,
  because the convention is the one thing the engine and the oracle share and
  vLLM does not: zero point (`q − 8` vs an explicit `qzeros`), the group axis,
  the `[K/8, N]` packing and nibble order, `desc_act`/`g_idx`. Do not touch a
  kernel first. *(Did not happen - no divergence to attribute.)*
- **all three agree** → the chain is closed. **← this one, 96/96.**

### How it was run

`tools/oracle/vllm_check.py` is the committed script. It reads the three
committed `.ids` files and hands them to the offline `vllm.LLM` as
`prompt_token_ids`, so **no tokenizer is in the loop** on either side and a
tokenizer difference cannot masquerade as a token difference. It reads the
oracle's `tokens` tensor (`i32[32]`) straight out of
`oracle-out/<p>.golden.safetensors` - the golden file itself, not a derived
copy - and prints the comparison. Exit 0 iff every id matched.

Two settings matter for the check to mean what it says.
`SamplingParams(temperature=0, max_tokens=32, min_tokens=32, ignore_eos=True)`:
`ignore_eos` is there because `dump.py`'s greedy loop is a bare argmax with no
stop condition, so without it a stop would silently shorten the comparison. And
`enable_prefix_caching=False` with `max_num_seqs=1`, one `generate()` call per
prompt - a correctness check has no business letting a cached prefix or a
two-prompt batch change which kernel shape runs.

The container is the one recorded in `docs/BENCHMARKS.md` (the 31.50 t/s
baseline was measured with it, on this same checkpoint), run as the calling
uid/gid - the recorded gotcha; as root the outputs come back root-owned:

```bash
# on the box, from ~/b70-inference-server
docker run --rm --entrypoint bash -u "$(id -u):$(id -g)" \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --device /dev/dri \
  -v /dev/dri/by-path:/dev/dri/by-path:ro -v /sys/class/drm:/sys/class/drm:ro \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --ipc=host --pid=host --net=host --shm-size=16g \
  -e ZE_FLAT_HIERARCHY=FLAT -e ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_XPU_ENABLE_XPU_GRAPH=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e CCL_ZE_IPC_EXCHANGE=sockets \
  -e HF_HUB_OFFLINE=1 -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e HF_HOME=/scratch/hf -e HOME=/scratch -e PYTHONUNBUFFERED=1 \
  -v "$HOME/b70-inference-server:/ws" -w /ws \
  -v "$HOME/.cache/huggingface:/hf:ro" -v /tmp:/scratch \
  vllm-xpu-env-next-p314-t214-vxkp0:latest -c \
  'SNAP=$(ls -d /hf/hub/models--Vishva007--Qwen3.8-27B-W4A16-AutoRound-GPTQ/snapshots/*/ | head -1);
   python3 tools/oracle/vllm_check.py "$SNAP" \
     --prompts /ws/tests/golden/prompts --golden /ws/oracle-out'
```

### The ids

vLLM's 32 generated ids per prompt, copied out of the run's stdout. The
oracle/engine row is the golden `tokens` tensor, which the script read from the
`.safetensors` in the same process:

```
prose  (42 prompt ids)
  oracle/engine  3113 7810 279 1118 479 654 8980 1000 381 1142 440 279 1834 725 2213 13
                 3113 11292 279 4220 6092 1000 381 6992 11 321 539 5600 279 72103 1000 381
  vllm           3113 7810 279 1118 479 654 8980 1000 381 1142 440 279 1834 725 2213 13
                 3113 11292 279 4220 6092 1000 381 6992 11 321 539 5600 279 72103 1000 381
  MATCH 32/32

code   (61 prompt ids)
  oracle/engine  271 727 40523 17 19490 11 750 11 15131 1590 198 262 460 498 1030 8474
                 3620 11 750 681 15131 8 364 343 303 2663 60 271 727 40523 18 19490
  vllm           271 727 40523 17 19490 11 750 11 15131 1590 198 262 460 498 1030 8474
                 3620 11 750 681 15131 8 364 343 303 2663 60 271 727 40523 18 19490
  MATCH 32/32

cjk    (38 prompt ids)
  oracle/engine  29545 271 95815 108553 97663 108447 96494 3709 98844 95895 97771 95726 114183 101650 100700 1710
                 271 550 220 99737 96863 271 99737 96863 95761 105064 97463 95793 100830 98252 96019 115534
  vllm           29545 271 95815 108553 97663 108447 96494 3709 98844 95895 97771 95726 114183 101650 100700 1710
                 271 550 220 99737 96863 271 99737 96863 95761 105064 97463 95793 100830 98252 96019 115534
  MATCH 32/32

=== VERDICT: 96/96 ids equal across 3 prompts
=== ALL THREE AGREE - the trust chain is closed (docs/14 §cross-check).
```

vLLM's detokenisation of those ids is the same three continuations recorded in
`tools/oracle/README.md` - including the CJK prompt's completion of the trailing
emoji's variation selector, and `code`'s `def clamp2(values, lo, hi):` /
`    return [min(max(v, lo), hi) for v in values]`.

### Why this is a third implementation and not a second look at the same code

Nothing in the vLLM path is shared with the engine or the oracle. Read off the
run's own log: the architecture resolves to `Qwen3_5ForConditionalGeneration`,
the int4 weights go through **`XPUwNa16LinearKernel for AutoGPTQLinearMethod`**
- vLLM's own GPTQ unpack, not `dequant.py` - the GDN layers run the **Triton**
prefill and decode kernels (`qwen_gdn_linear_attn.py`, the fused CUDA path
declined on an XPU), and the full-attention layers run **FlashAttention v2**.
Different unpack, different kernels, different device (XPU vs the oracle's CPU
and the engine's own 774 Level Zero kernels), different scheduler. The only
thing all three share is the checkpoint's bytes and the prompt ids.

That is exactly what makes 96/96 informative: it is the packing convention -
zero point `q − 8` with no `qzeros` stream, group 64 along `K`, `[K/8, N]` u32
words, `desc_act: false` - being read the same way by an implementation that
never saw `dequant.py`. `dequant.py` is now cross-checked, not just
self-consistent, and it is **no longer a suspect** for a divergence measured on
these three prompts.

### Measured cost, and the control run

| Run | `enforce_eager` | load | prose | code | cjk | verdict | rc |
|---|---|---|---|---|---|---|---|
| compiled (the recorded serve config) | `False` | 212.1 s | 1.1 s | 1.1 s | 1.0 s | **96/96** | 0 |
| control | `True` | 57.1 s | 4.7 s | 4.7 s | 4.6 s | **96/96** | 0 |

The second run exists so the verdict does not rest on the torch.compile / XPU
Graph path. It is the **same command as above with `--enforce-eager` appended**
to the `python3 tools/oracle/vllm_check.py "$SNAP" …` line - same image, same
env, same mounts, same `-u $(id -u):$(id -g)` - and its log is
`~/b70-inference-server/oracle-out/vllm_check_eager.log` on the box (the
compiled run's is `vllm_check.log` beside it; both are under `oracle-out/`, so
neither is committed).

`--enforce-eager` takes a different execution path through the same weights and
produced **byte-identical id blocks**. That claim is one `md5sum` per log over
exactly the id lines of the three `=== <prompt>: 32 ids in …` stanzas - the
`grep -A2` takes each stanza header plus its two 16-id lines, and the second
`grep` drops the headers, leaving only the six lines of ids that are hashed:

```
$ cd ~/b70-inference-server/oracle-out
$ grep -A2 -E "^=== (prose|code|cjk): " vllm_check.log       | grep -E "^[0-9]+ " | md5sum
8decb0462f5af85ac0f0d6dbb63ae591  -
$ grep -A2 -E "^=== (prose|code|cjk): " vllm_check_eager.log | grep -E "^[0-9]+ " | md5sum
8decb0462f5af85ac0f0d6dbb63ae591  -
$ grep -A2 -E "^=== (prose|code|cjk): " vllm_check.log       | grep -E "^[0-9]+ " | wc -lc
      6     470
$ grep -A2 -E "^=== (prose|code|cjk): " vllm_check_eager.log | grep -E "^[0-9]+ " | wc -lc
      6     470
```

6 lines / 470 bytes on both sides, same digest: the eager run's 96 ids are the
same bytes as the compiled run's 96 ids quoted above, and therefore the same
96 golden ids. The eager run's three 32-id blocks are quoted verbatim in this
task's report (`.superpowers/sdd/2026-08-25-plan4-spec1.5-decode-performance/task-4-report.md`,
§fix round) rather than repeated here, because they are character-identical to
the block already above and a second copy would only be noise.

Model load is 17.07 GiB on device, 6.9 s of weight load; the rest of the
compiled run's 212 s is Inductor.

Both runs print `double free or corruption (fasttop)` from the XPU stack's
teardown **after** the verdict line and after `XPUWorker shutdown: done`, and
both still exit **0**. It is a shutdown artifact of the container's stack, it
happens once the comparison is already printed, and it is recorded here rather
than swept up because it is in the raw logs.

### Two caveats worth keeping

- **The box's `config.json` is locally edited.** sha256
  `a56f436ea109c6dc9ee707fee2e13933d2c3b87cb0ec7bbad8d1d5aef8b6579b`, 12440 B -
  the two `mtp` rules flipped to exclusions per `docs/BENCHMARKS.md` ("Checkpoint
  bug, not vLLM or XPU"). The edit only changes whether vLLM quantises the MTP
  head, which neither the engine nor the oracle loads at all (`dump.py`'s
  `SKIP_PREFIXES`). No quantised tensor either of them reads is affected. The
  script prints that hash on every run so a future re-run says whether it is
  comparing the same file.
- **Three prompts, 96 tokens.** This closes the chain *for these prompts*. It is
  a check with a recorded result, like the CLI cross-check above - not a
  regression test, and no new test binary was built for it. Re-run it after any
  change to `dequant.py` or the loader's packing assumptions.

## The RTN-checkpoint gate - 2026-08-26, and the tie it found

Everything above is the gate on `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`.
Spec 1.6 §5.1 added a second checkpoint -
`~/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64`, this project's own RTN
quantisation with a packed `lm_head` (docs/13) - and a golden set is bound to
exactly one checkpoint, so a second one was produced: **`oracle-out-rtn/`**,
same three prompts, same committed `.ids`, same `--gen 32`, same container.
The Vishva007 golden files were not touched.

### The standing gate did not move

Re-run 2026-08-26 with the whole spec-1.6 §5.1 change in the tree:

```
  prompt   ids   exact/32   tap min cos (layer,t)   L63 tail   gdn min cos   logit min cos
  prose     42   32/32      0.994806738 (62,24)   0.997115152  0.999851574 (L60)  0.999864052
  code      61   32/32      0.913533675 (59,41)   0.982882165  0.998530392 (L60)  0.999818731
  cjk       38   32/32      0.999690168 (51, 0)   0.999832133  0.999904153 (L33)  0.999876394
golden_gate_test OK: 3 prompts x 32 greedy tokens, element-exact against the CPU oracle
```

**96/96, and the full suite 40/40 green.** Nothing regressed.

### The new checkpoint's own gate: 79/96, and why

```
  prompt   ids   exact/32   tap min cos (layer,t)   L63 tail   gdn min cos   logit min cos
  prose     42   15/32      0.999817776 (62,41)   0.999842062  0.999900713 (L60)  0.628406785
  code      61   32/32      0.896471198 (63, 7)   0.788012934  0.999677518 (L60)  0.999830544
  cjk       38   32/32      0.999851232 (62,12)   0.999844791  0.999909240 (L33)  0.999936448
GATE FAILED: prose reproduced 15/32 tokens (first mismatch 15)
```

`code` and `cjk` are exact. `prose` diverges at generated step 15 and never
recovers - which is what a greedy sequence does after one different token, so
**15/32 is one event, not seventeen.** `prose`'s `logit min cos` of 0.628 is
that event's consequence: every row after step 15 compares two different
contexts. The cosine **at the decision row itself is 0.999944**, in the same
band as every other row in this document.

#### The decision row, read exactly

The oracle's own top-two logits at that row are **bit-identical**:

```
decision row for generated step 15 = logits[56]
   id    271  f32 19.750000000  bits 0x419E0000
   id    353  f32 19.750000000  bits 0x419E0000
   id   2717  f32 17.375000000  bits 0x418B0000
   top-2 gap = 0.000000000e+00   bf16 ulp at that magnitude = 1.250000000e-01
   BIT-EQUAL: True
```

The engine's fp32 logits separate the same pair by **0.0156, in favour of 353**
- which is **one eighth of one bf16 ulp** at that magnitude. `torch.argmax`
broke the exact tie by lowest index and returned 271.

**The oracle's logits are on the bf16 grid.** `dump.py` records
`out.logits[0].to(torch.float32)`, and `out.logits` from a bf16 model *is* bf16,
so widening it to fp32 lands exactly on bf16-representable values - every word
in the golden `logits` tensor has zero low 16 bits, verified over all six golden
files. The reference therefore carries ~8 mantissa bits at the decision, and
**cannot resolve any two candidates that fall inside one ulp of each other.**
This one is not merely inside an ulp; it is the same number.

#### How exposed each golden set is - measured over all six files

Exact top-two ties among the 32 **decision** rows:

| golden set | prose | code | cjk |
|---|---|---|---|
| `oracle-out/` (Vishva007) | 0 | 0 | **2** (steps 13, 26) |
| `oracle-out-rtn/` (RTN) | **3** (steps 15, 26, 29) | 0 | 0 |

**The published checkpoint's set has ties too, and the gate passes 96/96 there.**
On `cjk` steps 13 and 26 the engine happened to agree with torch's
lowest-index tie-break. So the standing 96/96 has, from the day it was recorded,
contained two decisions the reference does not determine - the coin came up
heads twice. That is a fact about the gate, not about either checkpoint, and it
is recorded here because it was not known before 2026-08-26.

#### The flip is one event, and that is measured, not argued

Teacher-forcing: feed the engine the prompt **plus the golden continuation
through the tied step** (42 + 16 ids) and generate the remaining 16.

```
engine  40 557 11362 383 279 172113 440 821 1142 2272 279 8981 9506 11 9799 279
golden  40 557 11362 383 279 172113 440 821 1142 2272 264 8981 9506 11 9799 279
```

**15 of 16**, and the single difference is at generated step 26 - the **second**
bit-exact tie (ids 264 and 279). At step 29, the third tie, the engine agrees
with the oracle.

The complete picture for `prose`, 32 decisions:

| | count | engine vs oracle |
|---|---|---|
| rows where the oracle's top-2 differ | 29 | **29/29 element-exact** |
| rows where the oracle's top-2 are bit-identical | 3 | 1 agree, 2 differ |

**The engine reproduces every decision the reference actually determines.**

#### What this is, and what is being asked

This is the flip spec 1.5's re-assessment memo §5.1 predicted in writing:

> "a flipped token here would be a *legitimate* flip, not a bug, and spec §2's
> ruling then applies: **stop and surface it as a decision**, never absorb it."

So it is surfaced, and **the gate has not been weakened.** `golden_gate_test`
still asserts exact token equality; the RTN checkpoint fails it at 79/96 and
will keep failing it until somebody decides what a gate should do at a row its
reference cannot resolve. The options, none of them taken here:

1. **Leave it.** The RTN checkpoint has no green gate. Honest, and it means the
   engine's correctness on that checkpoint rests on `code` + `cjk` (64/64) plus
   the 29/29 analysis above rather than on a passing test.
2. **Treat a bit-exact oracle tie as satisfied by either id.** Narrow, and
   arguably not a weakening at all - it declines to assert something the
   reference does not determine. It does change the gate's contract, it would
   also apply retroactively to `cjk`'s two rows on the published checkpoint, and
   it needs the tie detected from the golden `logits` at test time.
3. **Give the oracle more resolution** - have `dump.py` compute the head in
   fp32 so its logits stop landing on the bf16 grid. That changes the oracle's
   numerics, which is the one thing this document is most reluctant about, and
   it would invalidate both existing golden sets.
4. **Change the prompt.** Cheapest and worst: it hides a property of the gate
   behind a choice of input.

The third bullet of the trust chain (`tools/oracle/README.md`) already says the
engine is **the more accurate of the two** - it never materialises a bf16 weight
- and this row is that statement arriving somewhere it has consequences. A
reasonable reading of the measurement is that the engine is *right* at step 15
and the oracle is *tied*; that reading is not a licence to change the bar
without a ruling.

#### Diagnostics, RTN checkpoint, for the record

Per-prompt minima over the 64-layer tap, the L63 tail, the 48 GDN states and the
logit rows. The `code` prompt's low taps are the standing anomaly this document
already describes ("The `code` prompt's LOW rows"), and `code` is 32/32 exact
on this checkpoint too - low taps still do not predict flips.

| prompt | exact/32 | tap min cos | L63 tail | gdn min cos | logit cos at the decision rows |
|---|---|---|---|---|---|
| prose | 15/32 | 0.999817776 (L62, t41) | 0.999842062 | 0.999900713 (L60) | 0.99994 … 0.99998 for steps 0-15 |
| code | **32/32** | 0.896471198 (L63, t7) | 0.788012934 | 0.999677518 (L60) | 0.999830544 min |
| cjk | **32/32** | 0.999851232 (L62, t12) | 0.999844791 | 0.999909240 (L33) | 0.999936448 min |

`gdn_state`: 0 of 144 below the 0.999 diagnostic bar on any prompt.

#### The RTN golden set, re-read in a separate process

`OUT_DIR=oracle-out-rtn tools/oracle/check.sh`, 2026-08-26:

| prompt | bytes | tokens | distinct | `resid.L63` finite | continuation |
|---|---|---|---|---|---|
| prose | 311 030 280 | 32 | 25 | yes | ` By eight the whole town would be awake.\n\nI was not awake.\n\nI was sitting on the quay with my back against a cold stone, watching the` |
| code | 367 258 168 | 32 | 24 | yes | `def clamp2(values, lo, hi): return [min(max(v, lo), hi) for v in values]` then `def clamp3(values` |
| cjk | 299 192 864 | 32 | 29 | yes | `我站在船舷边，看着对岸的轮廓一点点清晰起来。` then a `## 渡轮，是这座城市的另一种呼吸` heading |

Two things worth noticing. **`code`'s 32 ids are identical to the published
checkpoint's** - an RTN quantisation and a tuned AutoRound one produce the same
continuation on structured text, which is a free signal that the RTN weights are
faithful. And `cjk` again opens by completing the trailing emoji's variation
selector, which is what that prompt exists to pin.

Regeneration cost, 2026-08-26, box under a 12-core vLLM XPU kernel compile,
`ORACLE_THREADS=28` (torch reported 22 intra-op threads - the image caps it):
**25 min 14 s** for all three serially, against 16 min 48 s on an idle box.

### Running the RTN gate

```bash
# the golden set (on the box, detached; ~25 min under load)
OUT_DIR=oracle-out-rtn \
ORACLE_SNAP=$HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64 \
ORACLE_THREADS=28 tools/oracle/golden.sh

# the gate itself: golden dir, prompt dir, snapshot
tools/box.sh run './build/tests/golden_gate_test "$PWD/oracle-out-rtn" \
  "$PWD/tests/golden/prompts" $HOME/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64'
```

`ctest` still runs the gate against `oracle-out/` and the published checkpoint -
that registration is unchanged, and it is the one that is green.

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
