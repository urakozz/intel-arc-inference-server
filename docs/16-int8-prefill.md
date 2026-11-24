# The int8 prompt pass

The prompt pass (prefill: reading the whole prompt before the first token is generated) runs in bf16 by default.
`--prefill-backend l0-int8` is an optional faster form that feeds every linear layer of the prompt pass int8
activations, which the card's matrix instruction multiplies at twice the bf16 rate. On Qwen3.8 it also keeps the
context (the KV cache) in int8 - it turns `--kv-cache int8` on - and computes the prompt's attention scores in int8.

This doc explains what runs in int8, how that recipe was chosen over the alternatives, what it gains and where it
fails. It is for readers deciding whether to turn the option on, and for anyone curious why a 4-bit model ends up
doing 8-bit arithmetic.

On Qwen3.8-27B, one card, served by `b70-serve` with MTP on, it reads a prompt 1.40x to 1.47x faster than bf16
(prompt tokens a second; the earlier int8 pass, int8 in the MLP only, for comparison). The bf16 pass runs
4096-row chunks at the default `--max-len auto`, which sizes its context for them; one run after a warm-up each:

| prompt tokens | `l0` (bf16) | earlier int8 pass | `l0-int8` | gain over bf16 |
|---:|---:|---:|---:|---:|
| 3,931 | 1,908 | 2,524 | 2,800 | 1.47x |
| 30,199 | 1,634 | 2,050 | 2,346 | 1.44x |
| 60,609 | 1,395 | 1,687 | 1,985 | 1.42x |
| 122,435 | 1,072 | 1,234 | 1,500 | 1.40x |

Two cards (`tp`, `pp`) run the same form; they have not been timed or gated with it yet (see "Tensor parallel").

What it costs, measured against the bf16 prompt pass on the same server: none of 12 long tool calls that copy exact
text (8k to 99k tokens) came out different; the most likely next token differed at 3 of 585 positions over 4k to
64k prompts (all three on 4k prompts); the golden gates passed 93 of 93 rows and the 2,820-token long gate 32 of 32
against the CPU reference. It stays opt-in because its rounding still drifts from the reference more than the bf16 path does; the
"Limits" section lists what it does not run with. On Ornith it is slower than the bf16 pass.

## What runs

```
stored once (the checkpoint)      int4 weights, one scale per 64 (g64)
                                        │
DECODE (unchanged)                 int4 → bf16 on the fly, bf16 math
                                        │
PREFILL, per linear layer, per chunk, on the fly:
  activations  x (bf16)  ──► rotate (Hadamard) ──► int8, one scale per token
  weights      int4 g64  ──► dequant ──► rotate (same Hadamard) ──► int8, one scale per column
                                        │
                           int8 × int8 on the XMX engines (2x bf16 rate)
                                        │
                           rescale → bf16/fp32 output (the rotation cancels exactly)
```

Nothing new is quantised. The file is the original `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`: W4A16 (4-bit
weights, 16-bit activations) is what is stored, and W8A8 is what prefill computes, rebuilt from the stored weights
on the fly for every 1024-column slab of every chunk. XMX is Intel's name for the matrix engines that run the DPAS
(dot-product-accumulate-systolic) instruction.

The rotation is R = D x blockdiag(H_1024) / 32: random signs times a Walsh-Hadamard transform over each block of
1024 channels. Activations in these models have a few outlier channels far larger than the rest, and one int8
scale per token would have to stretch to cover them, wasting precision on all the others. The rotation spreads the
outliers' energy over all channels, so int8 with one scale per token loses little. Because it is applied to both
sides, x R and R^T W, it cancels exactly: (x R)(R^T W) = x W.

The weight scales, one per output column of the rotated weights, depend only on the weights, so they are computed
once when the model loads. The recipe (Hadamard rotation, int8 math) is called "h8" in the code and below.

### Which linears run h8

On the dense model Qwen3.8 every prompt linear runs h8: the MLP's gate‖up and down, and the four mixer linears (the
gated delta-net's input and output projections, and attention's q‖k‖v and o_proj), all with the same 1024-wide
rotation. Only the gated delta-net's small a‖b projection and the output head stay bf16.

On one card (and in the two-card pipeline), a short chunk (640 rows or fewer) runs every linear in bf16 instead;
under `tp` it stays h8. The h8 path rebuilds every weight as rotated int8 on each chunk, whatever its length, so a
30-row chunk paid a full chunk's weight rebuild. The bf16 short-chunk GEMMs read the int4 weights directly and
spread the few rows over every core. The MLP of a 16-row chunk then takes 46 ms instead of 244 ms (all 64 layers),
and a 512-row chunk takes 239 to 256 ms instead of 257 to 268. At 768 rows the two paths tie, and above that h8 is
faster. The 640-row bound was measured on the MLP; the mixer linears follow it. This matters because the prefix
cache cuts prompts at 2048-token blocks, so most prompts end in a short chunk, and every follow-up turn of a session
is one.

The mixer linears were not always int8. An earlier build of this all-linears form made one row of the 2,820-token
long gate pick a token the f32 reference puts 7.9 logits below its own top, so for a while the pass ran int8 in the
MLP only and kept the mixer in bf16, at about two thirds of the speed-up. On today's kernels the all-linears form
passes that gate 32 of 32 and the measurements in "Quality" below, so it is the int8 pass again.

Mixture-of-experts models (MoE: each token is routed to a few of many expert MLPs) such as Ornith run no h8
linears at all. Under `l0-int8` their dense linears, router and GDN a‖b projection stay bf16 as on `l0`, and only
the routed experts take an int8 form: W4A8, the int4 weights against activations quantised to int8 per token and
per 64-wide group of k, with no rotation. The dense linears stay bf16 because an int8 dense linear could flip a
near-tie between two experts in the router.

A model whose hidden size is not a multiple of 1024 cannot use the 1024-wide rotation blocks, so Kolibri-1 and
Qwen3.8-Flash-Next (hidden size 2560) refuse `l0-int8` by name at startup.

## Why decode stays int4 and bf16

Decode multiplies one token's row by every weight matrix. With one row there is no matrix for XMX to work on, and
the multiply is not the bottleneck anyway: reading the weights from memory is. Decode speed is therefore set by
bytes per weight. The decode figure below for an 8-bit checkpoint is arithmetic from the measured int4 rate, not a
measurement:

| stored weights | bytes per weight | decode |
|---|---:|---|
| int4 g64 | ~0.53 | 1x |
| int8 or FP8 (a W8A8 / FP8 checkpoint) | ~1.0 | roughly half |

An 8-bit checkpoint would make decode slower, not faster, and the B70 has no FP8 matrix hardware anyway (Intel's
datasheet lists only INT8; the FP8 builtins are undeclared or crash the compiler). Decode gets faster from fewer
bytes, or from more tokens per weight read: MTP speculative decoding (a small head guesses several tokens and one
step checks them) or batching, both of which turn decode into a small GEMM that XMX can run.

## How the recipe was chosen

Thirteen alternatives were tried or costed. Accuracy figures are relative error against a higher-precision
reference; "end-to-end" means the model's logits over four test prompts.

| tried | result | verdict |
|---|---|---|
| FP8 / FP4 matrix math | not on this silicon: undeclared builtins, compiler crashes, INT8-only datasheet | impossible |
| int4 weights × int8 activations at the checkpoint's g64 (i8 × i4 DPAS) | 0.97x the bf16 control: the per-group rescale every 2 instructions eats the 2x | rejected on speed |
| the same at coarser groups | g128 1.22x, g256 1.43x, per-channel 1.75x (speed only) | only per-channel is fast |
| int8 activations without rotation (CPU, real layers) | gate‖up 2.8 %, down 5.1 % error; worst rows 0.998 / 0.957 cosine | rejected on accuracy |
| SmoothQuant instead of rotation | helps gate‖up, fails on down, where the outliers move from token to token | rejected |
| int4 activations (W4A4), rotated, g256 on both operands (CPU, whole model) | held-out 23.3 % relative L2 error against h8's 12.5 %; +7.7 to +13.6 points on four prompts, 3.4 argmax in 100 lost; power-of-two scales worse | rejected on accuracy |
| rotating the weights inside the engine: four variants, including an exact DPAS factorisation | the shared-memory butterfly was fastest; the DPAS form was exact but not faster | butterfly kept |
| a rotated checkpoint made offline (rotate bf16, then AutoRound) | the rotation is exact (3.9e-7) but makes the int4 weights ~25 % worse | rejected |
| per-channel int4, AutoRound on the rotated model | 22.9 % logit error against 7.77 % for g64 | rejected |
| the unrotated fast path (int8 activations and weights, no Hadamard) | 1.70x / 1.85x on gate‖up / down, but 17.8 % against 12.0 % end-to-end error | rejected |
| an 8-bit (W8A8 / FP8) checkpoint | would halve decode and take ~27 GB of the 32 (arithmetic, not built) | not pursued |
| runtime Hadamard + per-channel int8, W4A16 file unchanged (h8) | 12.5 % against 12.0 % end-to-end on four prompts; 1.35x to 1.52x per linear; 1.28x on the whole prefill | accepted |

Read together: the unrotated int8 and SmoothQuant rows say the activations need a rotation, the unrotated fast
path says the rotation cannot be skipped, the two rotated-checkpoint rows say it cannot be moved into the file,
and the in-engine butterfly made it cheap enough to do at run time.

h8 was once the default prefill. The bf16 path became the default for every model when its own kernels caught up,
and `l0-int8` became opt-in.

## What the speed is made of

These figures are from the first one-card build of the path, when every dense linear ran h8. They show where the
gain comes from, per fused linear, on a 2048-token chunk, paired against the bf16 slab walk of that time. The number
in brackets is how many of the model's 64 layers have that linear.

| linear (count) | bf16 ms | h8 speed-up | of which: rotating requant, GEMM (gate‖up) |
|---|---:|---:|---|
| gate‖up (64) | 6.083 | 1.369x | 1.32 ms, 2.66 ms of 4.44 |
| down (64) | 3.371 | 1.521x | |
| GDN qkv‖z (48) | 2.811 | 1.346x | |
| out_proj / o_proj (64) | 1.019 | 1.397x | |
| FA q‖k‖v (16) | 2.539 | 1.375x | |

The int8 GEMM itself runs at 264 to 297 TOP/s, 2.0x to 2.3x the bf16 GEMM. Part of that 2x is spent rebuilding
the rotated int8 weights every chunk: the requant is the one cost here that is not matrix math.

That first build read a 4,096-token prompt 1.28x faster than the bf16 prefill of the time, paired in the same
session, with decode unchanged. On the golden gate it matched the bf16 prefill apart from one bf16 near-tie, which
was accepted.

## Quality

Measured on Qwen3.8-27B, one card, against the bf16 prompt pass (`l0`) or the CPU reference:

| check | result |
|---|---|
| 12 served tool calls that copy exact text back, 8k to 99k tokens (MTP on, 256 tokens each) | all 12 byte-identical to `l0`'s |
| next-token choice after the prompt, then 64 forced decode steps, 9 prompts at 4k / 32k / 64k (the bf16 side on the bf16 cache) | 582 of 585 positions agree (99.5 %); mean KL 5.2e-3, at 32k and 64k 1e-5 to 2e-4 |
| golden gates (prose, code, CJK; also at chunk 16, chunk 1000, chunk cap 4096, int8 head) | 93 of 93 determined rows each, under the standing allowance of one near-tie |
| long gate, 2,820 tokens in chunks of 1,024 | 32 of 32 rows, worst layer cosine 0.982 |

The three positions that differ are all on 4k prompts; the MLP-only form differed on four. The int8 state takes
0.085 GB (every linear's rotated column scales), and the bf16 slab stays allocated for the short chunks.

## Tensor parallel

Under tensor parallel (TP: every layer split across both cards) the same 1024-wide rotation is used. Each card's
half of down has a raw K of 8704, which is not a whole number of 1024 blocks, so the activation and weight kernels
zero-extend it to K = 9216 before rotating and running the int8 GEMM. Checkpoint addressing keeps the raw K; only
the int8 scratch and the GEMM use the padded width. Padding with zeros leaves the dot product unchanged before
quantisation, and needs neither another crossing of activations between the cards nor a persistent weight copy.
The other per-card widths already fit whole blocks: out_proj / o_proj K = 3072 and the column-parallel linears
K = 5120, and every output width is whole 1024-column slabs. As on one card, every linear runs int8, the mixer
linears included. Each card keeps its half of the context in int8 and scores its 12 query heads against its 2 KV
heads in int8, with the same kernel built at the card's head counts; the pipeline (`pp`) runs the one-card form on
each card's layers.

A 512-wide local rotation was tried first and removed after f32-rejected differences of up to 30.67 bf16 ulps.
Rotating at full width before splitting would need activation and weight data from the other card at the split
boundary. Padding locally keeps TP's wavefront and the established rotation size.

`tp_prefill_int8_test` covers four golden and 36 tool-call prompts. It teacher-forces one card's bf16 ids and
judges differences on that same context against the f32 reference, with the documented near-miss bar of eight
bf16 ulps (`spec_judge.h`, [14-golden-gate.md](14-golden-gate.md)). Missing references print their context keys
and SKIP. `--one-card` runs the one-card int8 pass on the same contexts as a diagnostic.

Measured idle on both cards with an earlier build of this all-linears form, interleaved, median of 3, MTP off, with
2048-row chunks, int8 head, bf16 KV cache, `max_len` 32768 and 32 generated tokens:

| prompt tokens | TP bf16, t/s | TP int8, t/s |
|---|---:|---:|
| 4096 | 2934.67 | 3322.73 (+13.2 %) |
| 16384 | 2673.31 | 3047.73 (+14.0 %) |

On that build the TP gate's one differing row, `t2_explain-loader` row 32, picked a token 2.55 bf16 ulps below the
f32 top, inside the near-miss bar. The current build has not been timed or gated on two cards yet, under `tp` or
`pp`.

## Limits

On Qwen3.8 `l0-int8` runs on the int8 context: the CLIs turn `--kv-cache int8` on with it and refuse an explicit
`--kv-cache bf16`. The prompt's attention scores are int8 too: each query row is quantised to int8 per head with one
fp32 scale over its 256 rotated dims, and Q times K is an int32 dot product scaled by the query's and the key's
scales; softmax and the value product stay as they are. Half the bytes of Q and K reach the matrix engines, which is
where the long-prompt attention kernel spends its time, so the gain grows with the prompt. Short chunks deep in a
session keep bf16 scores. The measured cost is the table above; with the MLP alone in int8 and the int8 cache, about
3 % of exact-copy tool calls once came out wrong, which this form, measured, does not repeat.

Every time `l0-int8` is on, the server prints a startup notice with the measured cost above.

On an MoE model, int8 only in the routed experts carries some quality risk, since a near-tie between two experts can
flip, and on Ornith it is slower than the bf16 pass. More generally the int8 path's results drift further from the
reference than bf16's, which is why it is never the default.

Everything else works on either prefill backend: flash attention and 128k context, prefix caching, MTP and the other
proposers (DFlash, prompt lookup), and `--parallel pp`, which splits the prefill across both cards. On one card Qwen3.8 prefills in 4096-row chunks where the memory plan has room, with 2048-row
chunks over a prompt's last 6,144 to 8,191 positions; that applies to both backends.

Short prompts have their own kernels on both backends: prefix caching and the short-chunk kernels remove the fixed
cost a short prompt once paid (see [12-kernels.md](12-kernels.md), "Short chunks").
