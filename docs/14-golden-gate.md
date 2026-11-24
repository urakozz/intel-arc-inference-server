# The golden gate: how correctness is proven

This document explains how the engine's output is checked against a reference it did not write, what the check
asserts and what it only reports, and the rules for the cases where two correct implementations legitimately
disagree. It is for anyone changing a kernel, the loader or the decode loop who needs to know what "it passes" means
here.

## Why a gate against an outside reference

Every kernel test in this repo compares a kernel with a host reference this project wrote: `gemv_test` grades
`gemv.cl` against `gemv_ref.h`, and `gdn_step_test` grades `gdn_step.cl` against `gdn_ref.h`. Those tests find
transcription bugs. They cannot find a shared misunderstanding: a reference and a kernel written from the same wrong
reading of the model's source code agree perfectly with each other and disagree with the model.

The golden gate closes that gap. It compares the whole engine (774 kernel launches, the loader, the int4 repack, the
captured command list, the decode loop) with a reference the project did not write: Hugging Face `transformers` on the
CPU, with the pure-torch gated delta rule and `attn_implementation="eager"`, over the same checkpoint, dequantised by
the same convention the C++ loader is bit-compared against (`tools/oracle/README.md`). That CPU reference is called
the oracle below, and the tensors it dumps are the golden set.

## Where it stands

The record, on the bf16-head Qwen3.8 checkpoint, is 3 prompts of 32 greedy tokens each: 93 of 93 determined rows
exactly equal to the oracle, and 3 undetermined rows, all inside the oracle's own set of tied top choices
("Undetermined rows" below explains the term). The gate is rerun by the maintainer's validation runs, not on every change.

The prompts are 42, 61 and 38 token ids of English prose, Python source, and Chinese text ending in an emoji
variation selector. Their ids are committed (`tests/golden/prompts/{prose,code,cjk}.ids`). The roughly 1 GB of golden
`.safetensors` are not; `tools/oracle/golden.sh` regenerates them.

`prefill_gate_test` runs the same comparison through one `Engine::prefill` per prompt instead of one replay per id,
against the same oracle, and reaches the same tally. The same gate also runs with the int8 output head
(`golden_gate_i8head_test`; the oracle stays bf16) and with the int8 KV cache (`golden_gate_kv8_test`).

The other models have their own gates built on the same ideas, each against its own reference:
`golden_gate_ornith_test`, `kolibri_golden_test` and `qwen4exp_golden_test` (Flash-Next). The mixture-of-experts
gates (Ornith, Flash-Next) also compare which experts the router picks for each token. A routing decision can flip on
a near-tie, when the 8th and 9th scores are a hair apart, so a different expert set passes only when the reference's
own gap at the cut is a near-tie.

## What the gate asserts, and what it only reports

| | Role | Measured |
|---|---|---|
| generated ids against the golden `tokens`, per prompt | the gate, and the only thing that fails the test | 93/93 determined rows exact; 3 undetermined, all in the golden set of tied top choices |
| GDN state after the prompt, per layer, cosine | diagnostic, marked `**LOW**` below 0.999 | 0.999912369 (prose), 0.999712545 (code), 0.999901750 (cjk); 0 of 144 layer-prompt pairs below the bar |
| per-layer residual tap cosine | diagnostic, marked `**LOW**` below 0.999 | a run of low rows on `code`, down to about 0.989 at layers 52 to 57 |
| logits row at every decision point, cosine | diagnostic | above 0.9998 on every row |

GDN is the gated delta net, the linear-attention layer that makes up 48 of Qwen3.8's 64 layers; its recurrent state
is what it carries from token to token. A tap is a copy of the residual stream (the running hidden state each layer
adds to) taken after a layer.

Token equality is the gate. Every cosine in this document is a diagnostic: a low one is marked `**LOW**` in the log
and does not fail the run. This is the only arrangement that survives 64 layers of bf16 rounding drift. A hard bound
on a tensor is either too loose to be useful or close enough to flake, and the `code` prompt below shows it. The GDN
state cosine was first specified as a hard bound and was demoted to a diagnostic before it ever shipped: the margin was
5.4e-4 at the time, so one legitimate numerics change, or one longer prompt, would have failed the gate on a
recurrent-state cosine while every token id was exact.

### Undetermined rows

A golden decision row whose top-1 is not unique is undetermined. On such a row the gate requires the engine's id to be
one of the oracle's tied top choices. On every other row it asserts exact equality, and every such row must pass.

The rule exists because the oracle's logits lie on the bf16 grid. The dump records
`out.logits[0].to(torch.float32)`, and `out.logits` from a bf16 model is bf16, so widening it lands exactly on
bf16-representable values: every word in every golden `logits` tensor has zero low 16 bits. The reference therefore
carries about 8 mantissa bits at the decision and cannot tell apart two candidates that fall within one ulp (unit in
the last place) of each other.

It was found when one checkpoint's `prose` prompt diverged at generated step 15 and never recovered. That is what a
greedy sequence does after one different token: 15 of 32 is one event, not seventeen. The oracle's top two logits at
that row were bit-identical:

```
decision row for generated step 15
   id    271  f32 19.750000000  bits 0x419E0000
   id    353  f32 19.750000000  bits 0x419E0000
   top-2 gap = 0.000000000e+00   bf16 ulp at that magnitude = 1.250000000e-01
```

The engine's fp32 logits separated the same pair by 0.0156, about an eighth of one bf16 ulp, in favour of 353.
`torch.argmax` broke the exact tie by lowest index and returned 271. The gate had been grading the engine on
`torch.argmax`'s tie-break without anyone noticing, until a tie came up.

The rule is not a tolerance. It declines to assert an answer the reference does not contain, and nothing more. It was
applied to earlier gate runs as well: their tied rows were re-marked undetermined, and not one determined row changed.

`tests/golden/golden_gate_test.cc` computes the full set of tied top choices from the golden `logits` row at every
decision, so the count of ties is derived by committed code, not pasted from a console. One standing example: on the
`cjk` prompt, the decode path emits token 4960 at generated position 22 where the oracle emits 271; the oracle's own
logits mark that row undetermined because 4960 is among its tied top choices; and the prefill path takes 271 at the
same row. That is a sub-ulp tie between two independently rounded paths, confirmed by the oracle rather than argued.

### Teacher forcing, and one statement in the right place

After a divergence the gate continues teacher-forced: it feeds the oracle's token instead of the engine's, so the rest
of the sequence is still judged in the reference's own context rather than in a sequence that has already left it.

The forcing flag must be set before the step advances, not after. The replay at the divergence row is the one that
writes the diverging token into the KV cache, the convolution ring and the GDN recurrent state. If that replay
consumes the engine's own id and forcing starts only on the next row, every later row reads a contaminated state.
Measured both ways on the same prompt:

| flag set | teacher-forced rows' logit cosine | determined rows exact |
|---|---|---|
| after the advance | 0.628 to 0.998 | 26/29 |
| before the advance | 0.9999 | 29/29 |

Same code, one statement moved. The cosine band shows that the tail really is on the reference's context: 0.63 is
what a diverged sequence looks like, and 0.9999 is what every other row in this document looks like.

## How the tap is compared

The engine's per-layer tap is not layer *i*'s output. The copy runs after layer *i*'s last kernel, and the residual
stream is advanced only by the residual-fold kernel, so `tap[i]` holds the hidden state with layer *i*'s mixer (its
attention or GDN part) folded in and layer *i*'s MLP still waiting in the partials buffer, where layer *i+1*'s first
fold will pick it up. The oracle's `resid.L{i}` is the layer's output after the MLP. Comparing the two directly would
report a divergence that is only a difference in definition.

So the expected tap is built from two golden tensors:

```
expected_tap[i][t] = bf16( resid.L{i-1}[t] + mixer.L{i}[t] )     i >= 1
expected_tap[0][t] = bf16( embed[ids[t]]   + mixer.L0[t]   )
```

The sum is taken in fp32 from the two golden bf16 tensors and rounded to nearest even, because that is the arithmetic
the fold kernel does. The oracle dumps no embedding tensor, so layer 0's input is gathered on the host from the same
bf16 table the device gathers from, adding no error of its own.

Layer 63's post-MLP residual has no tap, and none is invented. It is covered twice instead: the residual buffer after
the final fence is that vector, so the test compares it with golden `resid.L63` directly, and the logits and token ids
cover it again.

## The two expected sources of divergence

Both were named in advance and both were measured.

### GEMV accumulation order and where the dequantisation rounds

The engine never materialises a weight. It computes `scale · Σ(q − 8)·x`, with the products and the sum in fp32 and
the scale applied once per group of 64. The oracle materialises `bf16(scale · (q − 8))` and lets torch multiply bf16
weights by activations. That is two accumulation orders and two rounding points on identical bits, and the engine is
the more accurate of the two. The analytic cost of the oracle's extra per-weight cast to bf16 is about 0.2% RMS per
weight (an 8-bit mantissa gives a relative rounding error of about 2⁻⁹); that figure is an estimate.

The effect on a real activation is measured. The embedding gather is an exact bf16 row copy, so all of the layer-0 tap
error comes from layer 0's mixer, and it measures 0.45% to 0.68% maximum relative error on one mixer output, the same
order as the estimate. A few thousand weight-level errors of that size accumulated into one dot product, then passed
through a norm and a second GEMV, landing at half a percent, is what an expected divergence looks like here.

### The softmax path

The oracle runs an fp32 softmax over the whole row. The engine runs a blocked flash-decode, keeping a running maximum
and sum per block and merging them in a reduction kernel. The `exp` implementations and the merge order both differ
from a single-pass softmax, so this was expected to make the full-attention layers measurably worse than the GDN
layers.

It does not. Comparing the mean of per-layer median cosines, the gap between the 48 GDN layers and the 16
full-attention layers is about 2e-6 on every prompt, three orders of magnitude below the diagnostic bar. The softmax
path adds no measurable divergence of its own at these depths, and the GEMV rounding is the whole story.

### A third thing that could have diverged, and did not

`lm_head` is `[5120][248320]` for tiling, but only 248077 of those columns are real tokenizer ids. The argmax kernel
masks the 243-column tail to `-INFINITY`, while the oracle takes the argmax over the full width. If a padding column
ever won a row, the two would disagree by construction. The test reports the golden argmax both masked and unmasked at
every decision row, and the two have been equal on every row measured.

## The `code` prompt's low rows

The `code` prompt is the only one with tap cosines below the diagnostic bar: a small number of (layer, position)
comparisons fall below 0.999, on three positions out of 61. Every layer's median over positions is about 0.99994, and
the other two prompts have no low rows.

At its worst, one position reads a tap cosine of 0.9863, and two facts rule out the cheap explanations. It is not a
small denominator: the oracle's magnitude there is 275.7, the middle of that layer's distribution, and the absolute
error is 58.8 against a typical 3. It is not one layer or one kernel: the error at that position grows smoothly with
depth (0.007 at layer 0, 1.017 at 40, 33.1 at 55, 58.8 at 59) and recovers in relative terms by layer 63, with no
step at any layer or layer kind. What it is: one token position whose residual direction the network amplifies
strongly, applied to the small GEMV rounding difference every position carries. Engine and oracle fall into the same
attractor; it is not a bug. The prompt's delta-net state cosines stay above the bar, the neighbouring positions are back
at about 0.9999, and all 32 of its generated ids are exact.

This is why the gate is token equality. A hard 0.999 tensor bound would have failed this run, on a prompt where the
engine reproduces the oracle's output perfectly. Tokens gate; tensors diagnose.

## Determinism

The gate has been run many times across many builds of the test binary, and every quantity two logs have in common
agrees to all nine printed digits. The diagnostics grew as they were added; they never moved.

That is `replay_determinism_test`'s property seen end to end on the real checkpoint: 774 kernels and 19 modules
captured, 8 tokens over 3 runs bitwise identical from a reset, on the token ids, on a per-layer residual tap and on
every persistent buffer, and once more from a re-zeroed state.

`prefill_determinism_test` does the same for the prefill walk: the same prefill twice from a reset leaves the GDN
state, the convolution ring, both KV caches, the control block and the current token byte for byte the same, at each
of three chunk widths. A third prefill after a reset, with the prefill scratch deliberately not re-zeroed, must give
the same bytes again, which proves no prefill kernel reads scratch it has not first written.

## The third implementation

The trust chain has a ceiling the gate alone cannot lift. The C++ loader and the oracle are pinned to the same
dequantisation convention, bit for bit, by `dequant_fixture_test`. So a convention that is wrong about how the
checkpoint's author packed the nibbles would move the engine and the oracle together, and the gate would still pass.

Closing that needs a third implementation that shares nothing with either. A one-off offline vLLM run was it. It
handed the three committed `.ids` files to an offline `vllm.LLM` as `prompt_token_ids`, so no tokenizer is involved on
either side and a tokenizer difference cannot pose as a token difference. It read the oracle's `tokens` tensor
straight from the golden `.safetensors`, not from a derived copy, and passed only if every id matched. The script is
not kept in the repo; the result below is the record.

The result: 96 of 96 ids equal across the three prompts. All three implementations agree.

Two settings make the check mean what it says. `SamplingParams(temperature=0, max_tokens=32, min_tokens=32,
ignore_eos=True)`, because the oracle's greedy loop is a bare argmax with no stop condition, and without `ignore_eos`
a stop would silently shorten the comparison. And prefix caching off with `max_num_seqs=1`, because a correctness
check should not let a cached prefix or a two-prompt batch change which kernel shape runs.

It counts as a third implementation, not a second look at the same code, because of what the run's own log shows:
the int4 weights go through vLLM's own GPTQ unpacking, the GDN layers run Triton kernels, the full-attention layers
run FlashAttention, and all of it runs on the GPU, against the oracle's CPU and the engine's own Level Zero kernels.
Different unpacking, different kernels, different device, different scheduler. All three share only the checkpoint's
bytes and the prompt ids. That is what makes 96/96 informative: the packing convention (zero point `q − 8` with no
`qzeros` stream, groups of 64 along `K`, `[K/8, N]` u32 words, `desc_act: false`) is read the same way by an
implementation that never saw our dequantiser.

A control run with `--enforce-eager` takes a different execution path through the same weights and produced
byte-identical id blocks, so the verdict does not rest on vLLM's compiled path.

Two limits apply. The check closes the chain for these prompts; it is a check with a recorded result, not a regression
test, and it should be rerun after any change to the dequantisation convention or the loader's packing assumptions.
And a self-quantised checkpoint with an int4 `lm_head` cannot go through this route at all: vLLM picks its backend
from `quant_method` and sends such a checkpoint down a post-load path that fails before generating. That the packed
head is the cause is a plausible reading of vLLM's source, not a measurement, since the traceback never names the
failing module. That link of the trust chain stays open for packed-head checkpoints.

## A green gate is not evidence unless the run says which kernel ran

This is the most expensive lesson in this document. A record was once claimed for a delta-net scan kernel that had
passed its gates. It had not run. The selector resolved, the gates ran, everything was green, and the kernel the run
actually launched was the old one.

Every prefill gate run now prints the scan entry it launched, read back from the pointer the launcher handed the
kernel cache after the walk, and the tests assert that it equals the entry the walk binds:

```
gdn scan entry LAUNCHED: pf_gdn_scan_dpas_split
```

Run with its dispatch proven, that same kernel failed: 92 of 93 rows, with one state cosine at 0.996. The cause was
one operand of one term, and the fix took it to 93/93. The whole episode is in
[12-kernels.md, "How the prefill kernels got here"](12-kernels.md#how-the-prefill-kernels-got-here).

Probes carry the same rule. The W4A8 probe proves its dispatch twice: once by asking the driver for the name of the
kernel handle the launcher received, and once by having the kernel write a signature word from the device.

## One declared approximation

The delta-net scan runs approximate split-bf16 arithmetic (each fp32 value carried as a high and a low bf16 part, so
the matrix engines can multiply it) and is gated on tokens and state cosines rather than on `memcmp`. It is the only
default in the prefill path that is not bit-identical to what it replaced, and it is worth 2.33x on its own profile
row. Every other kernel that replaced another is compared bitwise, not by tolerance.

Its known limit: one operand inside the scan's second correction term is still a single bf16 value, measured harmless
on three prompts of one checkpoint and not proven in general.

## When the kernel path differs: the f32 judge

Bit-for-bit comparison is only fair where the code path is unchanged. The engine matches a golden set bitwise wherever
it runs the same arithmetic as before. When a feature must use a different kernel, two correct bf16 implementations
differ at the last rounding, and greedy decoding flips on a near-tie (two candidate tokens whose scores are a hair
apart). The verify step of speculative decoding is such a case. Speculative decoding has a cheap proposer guess the
next few tokens and the model check them all in one pass of several rows; those rows run the small-batch GEMM
`gemm_sm` rather than the one-row GEMV.

For those cases the gate is the f32 judge (`tests/server/spec_judge.h`, built on `tests/golden/forced_judge.h`). The
test writes out each context it would judge (`B70_DUMP_FORCED`), a CPU job runs a float32 reference on exactly that
context (`tools/oracle/qwen_forced.py`), and the test reads it back (`B70_FORCED_JUDGE`). A speculative stream
passes when one of these holds:

- it is identical to the plain stream;
- it is no further from the f32 reference than plain decoding on the same context, meaning the judge rejects no more
  of its ids;
- every miss is a near-miss (`golden::NearMiss`): the speculative id is at most `kNearMissUlp` = 8 bf16 ulps below
  the f32 top choice.

For logits rows, the verify row's cosine distance to f32 may be at most `kCosSlack` = 2 times the plain row's. A real
kernel error looks very different. The DPAS verify-attention kernel (DPAS is the Xe matrix-engine instruction) once
put its id at f32 rank 3, 4.0 below a top whose margin was 3.44, and it was switched off until fixed. The MoE golden
gates have their own, tighter tie rule: candidates within one bf16 ulp of the top count as a tie
(`golden_common.h`).

Three further rules keep the judge honest without loosening its bars.

The bf16-unstable rule. At a row where standard bf16 itself lands far from f32 (its gap between the two ids differs
from the f32 gap by more than 8 ulps, read from a transformers bf16 reference, `tools/oracle/qwen_bf16_top.py`), a
miss is graded against how far plain decoding already is from f32 on that pair, not against the fixed 8 ulps. Every
row graded this way is printed with its three gaps.

Reviewed rows. A few chaotic rows at turn boundaries fail every rule although nothing is wrong: rounding amplified
over 64 layers. Each is an explicit entry in `kReviewedRows`, keyed by the judge's context key, the row and the two
ids; only an exact match passes, and it is printed loudly as REVIEWED. The longer-context variants of the same tests
match by test tag, row, ids and the f32 gap (within 0.01 ulp). An entry that no miss matched is printed as stale and
can be removed.

Awaiting a reference. A divergence the plain rules fail, in a context that has no f32 reference yet, is not judged
and so not failed either: it is reported as "awaiting f32 reference", and the test is skipped (exit code 77) when
nothing else failed. A skip is never a pass. The contexts are dumped as always, and the f32 reference job turns the dumps into
references. Every divergence a judge does grade, and every structural or bitwise check, fails as before.

New references are made quickly with the CPU bf16 shim (`tools/oracle/cpu_bf16.py`). The test machine's CPU has no bf16
hardware, and torch's emulated bf16 GEMM there is about a thousand times slower than float32, so the shim does the
GEMM in fp32 and rounds once to bf16. It is not bitwise torch's (`--no-exact-bf16-gemm` reproduces the old path and
the existing golden sets), but it is as close to the f32 judge. CPU reference jobs run one at a time.

## Running the gate

```bash
ctest --test-dir build --output-on-failure -R golden_gate_test
# or, for the full diagnostic log:
./build/tests/golden_gate_test "$PWD/<golden-dir>" "$PWD/tests/golden/prompts" <snapshot>
```

A golden set belongs to exactly one checkpoint, and to the container image that produced it, since the oracle is
`transformers` running inside that image. `ctest` registers one golden directory; other sets are run by hand with the
golden directory, the prompt directory and the snapshot as arguments.

Missing golden files make the test exit 77, which ctest reports as skipped. The test also checks that each `.ids`
file has as many ids as its golden file's first residual tensor has rows, so a stale ids file cannot pass silently.

On an earlier build the gate took about 24 s standalone, 13.4 s of it the checkpoint load, at a peak resident memory
of 19.5 GiB with a bf16 head. One checkpoint load serves all three prompts. Regenerating a golden set took 18 min 16 s
for all three prompts on an idle machine, and 28% longer beside a 12-core compile: the CPU oracle is the one thing in
this project that really competes for the host.

`ctest -LE checkpoint` runs the tests that need no model files, as a quick check of a branch; it does not gate a
merge. Count passed, failed and skipped tests separately, because a skipped test, such as the gate with no golden set
present, is not a pass. The tests that need checkpoints and golden sets run in the maintainer's validation runs, on
demand, and a test whose inputs are byte-identical to a recorded pass keeps that pass.
