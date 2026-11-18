# Weight and activation formats

This document explains which number formats the engine stores weights in and computes activations in, and why.
It is for a reader who wants to understand the trade between bytes, accuracy and hardware support that sets every
model's on-disk and on-card form. The hardware facts it relies on are measured in
[01-hardware.md](01-hardware.md).

The decision: weights are int4, activations stay 16-bit (W4A16), and every supported checkpoint is normalised to
one canonical on-card layout at load time.

## Why not the alternatives

| Candidate | Verdict |
|-----------|---------|
| FP8 | Rejected. 8 bits a weight is twice int4's traffic on a decode that is bandwidth-bound, and there is no reachable FP8 matrix multiply on this card at all: the compiler declares only the scaled forms and the backend crashes on them ([01-hardware.md](01-hardware.md)). Its whole advantage elsewhere is hardware this card lacks. |
| MXFP4 | Rejected on this card. The backend refuses FP4 DPAS by name and every microscaling entry point crashes the compiler, so MXFP4 would have to be emulated as int4 plus a decode in the ALU, with block scales applied outside the instruction. At an effective 4.25 bits a weight it is also byte-identical to GPTQ g64, so it buys nothing for a bandwidth-bound decode either. |
| int4 GPTQ | Supported; the same tensor layout as AutoRound (below). |
| int4 AutoRound | The primary format: the best accuracy at 4 bits of the options we have checkpoints for. |
| int4 compressed-tensors (llm-compressor `pack-quantized`) | Supported when symmetric, group 64 or 128. The nibbles are GPTQ's (q + 8, low nibble first), stored [N][K/8], with scales [N][K/g]. The loader transposes both into the GPTQ layout exactly and needs no kernel change ([13-loader.md](13-loader.md), "compressed-tensors symmetric checkpoints"). Asymmetric checkpoints, activation-order permutations and other group sizes are refused. Loading one prints a note that our AutoRound g64 is the recommended format. |

## AutoRound and GPTQ: a config difference, not a layout difference

GPTQ and AutoRound are two offline quantisers: programs that turn a 16-bit checkpoint into int4 weights plus a
scale per group of weights. Checkpoints in this family declare something like:

```
quant_method    "auto-round"
packing_format  "auto_round:auto_gptq"
bits            4
sym             true
group_size      64
```

The packed nibble layout is GPTQ's. Two things differ. The offline rounding algorithm (AutoRound's signed-gradient
search, GPTQ's Hessian-based rounding) changes the weight values, not how they are arranged. And the metadata:
AutoRound adds `extra_config` and `dynamic` per-layer overrides that let one checkpoint mix bit widths.

So one unpacking routine covers both formats, and only the metadata parser differs. It must accept two spellings
of the same thing. The Qwen3.8 checkpoint says `quant_method: "gptq"`, `provider: "auto-round"`,
`desc_act: false`, `group_size: 64`, `sym: true`, `lm_head: false`; others say `quant_method: "auto-round"` with
`packing_format: "auto_round:auto_gptq"`. The nibbles on disk are the same. The loader keys on the tensor suffixes
(`qweight`, `scales`, `qzeros`, `g_idx`), never on the label.

What GPTQ packing ships per linear layer, read from the headers of `in_proj_qkv` (K=5120, N=10240, g64):

| Tensor | dtype / shape | Bytes | Read at decode? |
|---|---|---|---|
| `qweight` | I32 `[K/8, N]` = `[640, 10240]` | 26.2 MB | yes; this is the traffic |
| `scales` | F16 `[K/64, N]` = `[80, 10240]` | 1.6 MB | yes; 6.25 % on top |
| `qzeros` | I32 `[K/64, N/8]` = `[80, 1280]` | 0.4 MB | no: `sym: true` makes every zero point the constant 8, so it is dropped at load |
| `g_idx` | I32 `[K]` | 20 KB | no: `desc_act: false` makes it the identity, so the loader asserts that and drops it |

Across the whole model `qzeros` and `g_idx` are 0.20 GB of 15.7 GB, a free 1.3 % that a kernel reading the GPTQ
layout verbatim would pay on every token. AutoRound also ships an AWQ-packed variant (`auto_round:auto_awq`) whose
nibble order is `[0,2,4,6,1,3,5,7]`. The loader does not convert it: any packing other than
`auto_round:auto_gptq` is refused by name.

### Is AutoRound still needed?

The algorithm, yes; its runtime, no. AutoRound is an offline quantiser that produces better int4 weights than
rounding to nearest, and every checkpoint the engine serves is made with it. What the project drops is ARK
(`auto_round_kernel`), AutoRound's runtime `woqgemm` library; the engine has its own kernels. Keep the recipe,
discard the oven.

Which AutoRound build each quantise script pins, why, and which version made each published checkpoint is in
`tools/quantize/README.md`, "AutoRound version". The `auto_round:auto_gptq` config keys and packing above are the
same in 0.14.2 and 0.17.0.

## Mixed bit width is a real hazard

AutoRound's `dynamic` rules let one checkpoint carry layers at different widths. One published 27B has 97 layers
at bf16, 17 at int8 and the rest int4; the int8 ones are the attention `out_proj` and `o_proj`, two `down_proj`,
and layer 0's `in_proj_qkv` and `in_proj_z`.

The loader therefore treats the width of each layer as data, not as a global constant. Retrofitting that later is
painful; vLLM's XPU path still hard-gates it and raises `NotImplementedError` on 8-bit.

The Qwen3.8 checkpoint exercises this mildly. Its 98 `dynamic` exclusion rules leave `linear_attn.in_proj_a` and
`in_proj_b` in bf16 in all 48 GDN layers (GDN, gated delta net, is the model's linear-attention layer;
these two are its gate and decay projections, `[48, 5120]` each, 0.5 MB), plus the whole MTP head in bf16
(0.85 GB; MTP, multi-token prediction, is a small extra layer that drafts future tokens). Everything else is int4.
The loader reads the exclusion list and the tensor suffixes and assumes nothing.

## Format plurality is a load-time problem

All supported formats normalise into one canonical on-card layout while the model loads. That has three
consequences. The decode kernel knows one layout, so there is no branching at run time: no dispatch on
`packing_format`, no `is_awq_packed` checks in the hot path. Pre-swizzling, rearranging weights once at load into
whatever order the kernel's inner loop wants, is free. Every CUDA backend does this; vLLM's XPU MXFP4 path notably
does not (`compressed_tensors_moe_w4a4_mxfp4.py` has a bare `pass` where the repack would go), which is
measurable headroom left unused. And adding a format is a loader change, not a kernel change.

The canonical int4 layout is "layout 1", the second of two candidates described in
[12-kernels.md](12-kernels.md) ("layout 0" is plain row-major): a tiled 544-byte block per 16 `n` by 64 `k`, with
the scales inline. The choice was made by measuring both layouts at all five decode shapes, and it is close:
2713 against 2667 GB/s summed, within 2 % by the deciding metric. The whole lead comes from one shape, `qkv‖z` at
N = 16384, where layout 0's power-of-two 64 KB row stride caps it at 419 GB/s against layout 1's 559. On the other
four shapes layout 0 is 1.1 % to 6.8 % ahead. It is worth re-running if the shape mix changes. The full table and
the argument are in [12-kernels.md](12-kernels.md).

## Activations: W4A16 for decode and prefill, W4A8 measured and rejected

The weight format is settled: int4, symmetric. The width of the activations is a separate decision, and the
obvious split between the two regimes, decode (generating one token at a time) and prefill (reading the whole
prompt at once), turned out not to pay.

| | Activations | Why |
|---|---|---|
| Decode | bf16 (W4A16) | Bandwidth-bound. The multiply is not the cost and the activations are one vector, so quantising them buys nothing. |
| Prefill | bf16 (W4A16) | Compute-bound, and the obvious int8 candidate was built and measured (below). |

W4A8 was the natural candidate for prefill, and it lost. Battlemage does have a native mixed int8 x int4 DPAS
with no upconversion, gated only on the int4 matrix having no zero points, which `sym: true` satisfies. But the
mixed builtin's K is 32, the same as int8's, so it runs at exactly the int8 rate, 2.000x bf16 and not 4x
([01-hardware.md](01-hardware.md)). Built against real weights and real activations, the kernel measured 1.017x the
bf16 two-pass control, and 0.965x once the activation quantiser it needs was counted, against a bar of 1.53x: the
per-group rescale cost 3.0 to 3.4 ms of the 5.440 ms while the matrix math it rescales cost 1.99 ms. It also
showed 2.79 % relative L2 error against the bf16 result (worst-row cosine 0.9976), driven by activation outliers.

So the same checkpoint serves both regimes, and the activation path is the same too. Symmetric checkpoints are
still preferable, since an asymmetric one forfeits the mixed DPAS path outright and needs a zero-point stream in
every kernel, but that preference no longer rests on a performance claim.

## Group size costs traffic

Scales are read alongside the weights, so the group size, the number of weights that share one scale, is not
free:

| Scheme | Bits per weight (weights + fp16 scales) |
|--------|------------------------------------------|
| int4, g128 | 4.125 |
| int4, g64 | 4.25 |
| int4, g32 | 4.5 |
| MXFP4 (e8m0 per 32) | 4.25 |
| FP8 | ~8 |

g32 costs 9 % more traffic than g128 for accuracy that may or may not matter on a given model, so it is worth
measuring per model rather than assuming smaller is better. This project runs g64: finer scales for about 3 % more
bytes than the g128 default, and it matches a tile 64 elements deep in K with the f16 scales inline. Pick the group
size your kernels' tile geometry wants.

## What the engine serves today

Every supported model runs from an int4 checkpoint in this one format family: AutoRound, GPTQ packing,
symmetric, group 64, re-quantised by us (Qwen3.8, Ornith, Kolibri-1). The loader also reads the g128 spelling
(Intel's interim Flash-Next checkpoint) through an exact expansion to g64.

On top of that format, three choices are made per run, not per file. The output head (`lm_head`) is quantised at
load to int8 with one scale per row. The KV cache, the stored keys and values of every earlier token, is bf16 or,
with `--kv-cache int8`, a rotated int8 form. And the prefill linears are bf16 or, with
`--prefill-backend l0-int8`, the Hadamard-rotated int8 path of [16-int8-prefill.md](16-int8-prefill.md), computed
from the same int4 file. The int8 prefill together with the int8 KV cache is a known fail: the engine warns at
startup, and about 3 % of tool calls that copy exact text from a long prompt come out wrong
([low-priority-work.md](low-priority-work.md)).

MXFP4 and FP8 stay rejected for the reasons above, and no other third-party quantisation format is supported.
