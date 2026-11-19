# The int8 prompt pass

The prompt pass (prefill: reading the whole prompt before the first token is generated) runs in bf16 by default.
`--prefill-backend l0-int8` is an optional faster form that feeds some of the linear layers int8 activations,
which the card's matrix instruction multiplies at twice the bf16 rate.

This doc explains what runs in int8, how that recipe was chosen over the alternatives, what it gains and where it
fails. It is for readers deciding whether to turn the option on, and for anyone curious why a 4-bit model ends up
doing 8-bit arithmetic.

On Qwen3.8-27B it reads a prompt about 1.3x faster than bf16 on one card and about 1.10x faster under tensor
parallel:

| setup | `l0` (bf16), 4k / 16k prompt | `l0-int8`, 4k / 16k prompt |
|---|---:|---:|
| one card | 1,788 / 1,624 t/s | 2,337 / 2,059 t/s |
| two cards, `tp` | 2,931 / 2,683 t/s | 3,217 / 2,937 t/s |

It stays opt-in because its rounding drifts from the reference more than the bf16 path does; the "Limits" section
lists the known failures. On Ornith it is slower than the bf16 pass.

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

On the dense model Qwen3.8, only the MLP runs h8: gate‖up and down, the two linears where K = 17,408 and most of
the gain is. The four mixer linears (the gated delta-net's input and output projections, and attention's q‖k‖v
and o_proj) run in bf16, exactly as on `l0`.

Why not the mixer too: an all-linears form was built and is faster, but on a 2,820-token prompt it made one row pick
a token the f32 reference puts 7.9 logits below its own top, which is not a near-tie. With the mixer in bf16 the
long-prompt gate passes 32 of 32 rows and the logits sit as close to f32 as `l0`'s do. The price is about a third of
the speed-up over `l0` (see "Mixer in bf16: the trade").

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

## Mixer in bf16: the trade

The MLP-only form against the all-linears form on one card, interleaved, median of 3, while the other card ran a
test:

| prompt tokens | `l0` | `l0-int8`, all linears | `l0-int8`, MLP only | MLP only vs `l0` |
|---|---:|---:|---:|---:|
| 4,096 | 1,739.9 | 2,256.3 | 2,082.5 | 1.20x |
| 16,384 | 1,568.7 | 1,980.5 | 1,844.6 | 1.18x |

It keeps about two thirds of the gain of the all-linears form (1.30x / 1.26x). The `l0` column is from that session;
the bf16 pass has since become faster (1,788 / 1,624 t/s), so the current gain is smaller, as in the table at the top.
Accuracy improved:
`prefill_gate_long_int8_test`, on a 2,820-token prompt, passes 32 of 32 rows with the worst logits cosine 0.947
against the CPU reference (`l0`: 0.935; the all-linears form 31 of 32 at 0.819), and the per-layer split test
against bf16 moves closer everywhere (logits 0.9998 against 0.9994 to 0.9997). On the short golden prompts one row
changed: CJK row 9 is again the one-ulp bf16 near-tie accepted for `l0-int8` earlier, so those gates pass 92 of 93
under the standing allowance of one. `l0` is bit for bit unchanged.

It also needs less memory for the int8 state (0.074 GB instead of 0.085, since the mixer linears need no rotated
column scales), but now needs the bf16 slab (0.036 GB) on `l0-int8` too.

## Tensor parallel

Under tensor parallel (TP: every layer split across both cards) the same 1024-wide rotation is used. Each card's
half of down has a raw K of 8704, which is not a whole number of 1024 blocks, so the activation and weight kernels
zero-extend it to K = 9216 before rotating and running the int8 GEMM. Checkpoint addressing keeps the raw K; only
the int8 scratch and the GEMM use the padded width. Padding with zeros leaves the dot product unchanged before
quantisation, and needs neither another crossing of activations between the cards nor a persistent weight copy.
The other per-card widths already fit whole blocks: out_proj / o_proj K = 3072 and the column-parallel linears
K = 5120, and every output width is whole 1024-column slabs. As on one card, only gate‖up and down run int8; the
mixer linears take the bf16 slab walk on both cards.

A 512-wide local rotation was tried first and removed after f32-rejected differences of up to 30.67 bf16 ulps.
Rotating at full width before splitting would need activation and weight data from the other card at the split
boundary. Padding locally keeps TP's wavefront and the established rotation size.

`tp_prefill_int8_test` covers four golden and 36 tool-call prompts. It teacher-forces one card's bf16 ids and
judges differences on that same context against the f32 reference, with the documented near-miss bar of eight
bf16 ulps (`spec_judge.h`, [14-golden-gate.md](14-golden-gate.md)). Missing references print their context keys
and SKIP. `--one-card` runs the one-card int8 pass on the same contexts as a diagnostic.

Measured idle on both cards, interleaved, median of 3, with 2048-row chunks, int8 head, bf16 KV cache,
`max_len` 32768 and 32 generated tokens. The "all linears" column is the earlier form:

| prompt tokens | TP bf16, t/s | TP int8, all linears, t/s | TP int8, MLP only, t/s |
|---|---:|---:|---:|
| 4096 | 2934.67 | 3322.73 (+13.2 %) | 3,217 |
| 16384 | 2673.31 | 3047.73 (+14.0 %) | 2,937 |

The MLP-only form is 0.967x / 0.962x of the all-linears form, measured against it directly in a separate session.
With it, the TP gate has one row that differs: `t2_explain-loader` row 32 picks a token 30.67 bf16 ulps below the
f32 top (the all-linears form picked another one, 2.55 ulps below). It was reviewed and accepted as a known
divergence, so the test prints it and does not fail on it. In the same session the one-card pass on the same
contexts (`--one-card`) went from 3 rejected rows with the all-linears form (two on the long prompt at 63.5 and
50.3 ulps, and 27.5 on `t2_explain-linear_l0`) to 1 (8.55 ulps, `t6_usage-step` row 61).

## Limits

`l0-int8` together with `--kv-cache int8` is a known failure. Each prefill attention layer then reads K and V rows
the chunk has just written in int8, and the int8 linears amplify that rounding: about 3 % of tool calls that copy
exact text from a long prompt come out wrong. The server prints a startup warning for this combination; the `l0`
prefill or the bf16 cache avoids it. A fix is known (read the chunk's own new rows from bf16) and not built.

On an MoE model, int8 only in the routed experts carries some quality risk, since a near-tie between two experts can
flip, and on Ornith it is slower than the bf16 pass. More generally the int8 path's results drift further from the
reference than bf16's, which is why it is never the default.

Everything else works on either prefill backend: flash attention and 128k context, prefix caching, MTP and the other
proposers (DFlash, prompt lookup), and `--parallel pp`, which splits the prefill across both cards. On one card Qwen3.8 prefills in 4096-row chunks where the memory plan has room, with 2048-row
chunks over a prompt's last 6,144 to 8,191 positions; that applies to both backends.

Short prompts have their own kernels on both backends: prefix caching and the short-chunk kernels remove the fixed
cost a short prompt once paid (see [12-kernels.md](12-kernels.md), "Short chunks").
