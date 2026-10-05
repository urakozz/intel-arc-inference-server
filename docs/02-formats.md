# Format decision

**Decision: int4 weight-only, 16-bit activations (W4A16), normalised to one
canonical on-device layout at load time.**

## Why not the alternatives

| Candidate | Verdict |
|-----------|---------|
| **FP8** | Rejected. 8 bits/weight is 2x int4 traffic on a decode that is bandwidth-bound, *and* there is no reachable FP8 matmul on this card at all - the compiler declares only the scaled forms and the backend crashes on them ([01-hardware.md](01-hardware.md)). Its entire advantage elsewhere is hardware this card lacks. |
| **OpenVINO IR** | Rejected as a runtime format. It is a graph-and-weights container tied to the OpenVINO runtime; adopting it means adopting their execution model, which is the thing this project exists to replace. Worth reading `openvino.genai` for its scheduling ideas, not for its file format. |
| **MXFP4** | Rejected on this card. FP4 DPAS is refused by name by the backend and every microscaling entry point crashes the compiler, so MXFP4 would have to be emulated as int4 plus an ALU decode and out-of-instruction block scales. At 4.25 bits/weight effective it is also byte-identical to GPTQ g64, so it buys nothing for a bandwidth-bound decode either. |
| **int4 GPTQ** | Supported - same tensor layout as AutoRound (below). |
| **int4 AutoRound** | **Primary.** Best accuracy at 4 bits of the options we have checkpoints for. |
| **int4 compressed-tensors** (llm-compressor `pack-quantized`) | Supported when **symmetric**, group 64 or 128: the same nibbles as GPTQ (q + 8, low nibble first) stored [N][K/8], with scales [N][K/g]. The loader transposes both into the GPTQ layout exactly and needs no kernel change ([13-loader.md](13-loader.md), "compressed-tensors symmetric checkpoints"). Asymmetric checkpoints, activation-order permutations and other group sizes are refused (spec 20 §9). Loading one prints a note that our AutoRound g64 is the recommended format. |

## AutoRound vs GPTQ: a config difference, not a layout difference

Checkpoints in this family declare something like:

```
quant_method    "auto-round"
packing_format  "auto_round:auto_gptq"
bits            4
sym             true
group_size      64
```

The packed nibble layout **is** GPTQ's. What differs:

- the **offline rounding algorithm** (AutoRound's signed-gradient search against
  GPTQ's Hessian-based rounding) - it affects weight values, not their
  arrangement;
- the **metadata**: AutoRound adds `extra_config` / `dynamic` per-layer
  overrides that let a single checkpoint mix bit-widths.

So **one unpacking routine covers both formats.** Only the metadata parser
differs, and it must accept two spellings of the same thing. The gate checkpoint
says `quant_method: "gptq"`, `provider: "auto-round"`, `desc_act: false`,
`group_size: 64`, `sym: true`, `lm_head: false`; others say
`quant_method: "auto-round"` with `packing_format: "auto_round:auto_gptq"`. Same
nibbles on disk. The loader keys on the tensor suffixes
(`qweight`/`scales`/`qzeros`/`g_idx`), never on the label.

What GPTQ packing ships per linear, from the headers (`in_proj_qkv`, K=5120,
N=10240, g64):

| Tensor | dtype / shape | Bytes | Read at decode? |
|---|---|---|---|
| `qweight` | I32 `[K/8, N]` = `[640, 10240]` | 26.2 MB | yes - the traffic |
| `scales` | F16 `[K/64, N]` = `[80, 10240]` | 1.6 MB | yes - 6.25% on top |
| `qzeros` | I32 `[K/64, N/8]` = `[80, 1280]` | 0.4 MB | **no** - `sym: true` makes every zero point the constant 8; drop at load |
| `g_idx` | I32 `[K]` | 20 KB | **no** - `desc_act: false` makes it the identity; assert and drop |

Across the whole model `qzeros` + `g_idx` are 0.20 GB of 15.7 GB, a free 1.3%
that a kernel reading the GPTQ layout verbatim pays every token. AutoRound also
ships an AWQ-packed variant (`auto_round:auto_awq`) whose nibble order is
`[0,2,4,6,1,3,5,7]`; converting it to GPTQ order is a lossless permutation and a
load-time concern.

### Is AutoRound still needed?

**Yes, the algorithm. No, the runtime.**

AutoRound is an offline quantiser that produces better int4 weights than
round-to-nearest. Keep using it, or consume checkpoints others produced with it.
What this project drops is **ARK** (`auto_round_kernel`), its runtime `woqgemm`
library. We write that ourselves.

Clean separation: keep the recipe, discard the oven.

## Mixed bit-width is a real hazard

AutoRound's `dynamic` rules let one checkpoint carry layers at different widths.
One published 27B has **97 layers at bf16, 17 at int8, the rest int4**, the int8
ones being attention `out_proj`/`o_proj`, two `down_proj`, and layer 0's
`in_proj_qkv`/`in_proj_z`.

The loader must therefore treat **per-layer width as data, not as a global
constant**, from day one. Retrofitting that is painful; vLLM's XPU path still
hard-gates it and raises `NotImplementedError` on 8-bit.

The gate checkpoint exercises this in a mild form: its 98 `dynamic` exclusion
rules leave `linear_attn.in_proj_a` / `in_proj_b` in **bf16** in all 48 GDN
layers (`[48, 5120]` each, 0.5 MB - the gate and decay projections), plus the
whole MTP head in bf16 (0.85 GB). Everything else is int4. The loader reads the
exclusion list and the tensor suffixes; nothing is assumed.

## Design principle: format plurality is a load-time problem

All supported formats normalise into **one canonical on-device layout** during
model load. Consequences:

- **Zero runtime branching.** The decode kernel knows one layout. No dispatch on
  `packing_format`, no `is_awq_packed` checks in the hot path.
- **Pre-swizzling is free.** Rearrange weights once at load into whatever order
  the mainloop wants. Every CUDA backend does this; vLLM's XPU MXFP4 path
  notably does not (`compressed_tensors_moe_w4a4_mxfp4.py` has a bare `pass`
  where the repack would go). That is measurable headroom sitting on the floor.
- **Adding a format is a loader change**, not a kernel change.

**The canonical int4 layout is layout 1**: a tiled 544-byte block per 16 `n` by
64 `k`, scales inline. The decision was made by measuring both candidate layouts
at all five decode shapes, and it is close - 2713 against 2667 GB/s summed, so
the two are within 2% by the deciding metric. The whole lead comes from one
shape, `qkv‖z` at N = 16384, where layout 0's power-of-two 64 KB row stride caps
it at 419 GB/s against layout 1's 559; on the other four shapes layout 0 is
1.1% to 6.8% ahead. Worth re-running if the shape mix changes. The full table
and the argument are in [12-kernels.md](12-kernels.md).

## W4A16 for decode, and W4A8 for prefill was measured and rejected

The *weight* format is settled (int4, symmetric). The **activation** width is a
separate decision, and the obvious asymmetry between the two regimes turned out
not to pay off.

| | Activations | Why |
|---|---|---|
| Decode | bf16 (W4A16) | Bandwidth-bound. The multiply is not the cost and activations are one vector, so quantising them buys nothing. |
| Prefill | bf16 (W4A16) | Compute-bound, and the obvious int8 candidate was built and measured. See below. |

**W4A8 was the natural candidate and it lost.** Battlemage does have a native
mixed s8 x s4 DPAS with no upconversion, gated only on the int4 matrix having no
zero points, which `sym: true` satisfies. But the mixed builtin's K is 32, the
same as int8's, so it runs at exactly the int8 rate - 2.000x bf16, not 4x
([01-hardware.md](01-hardware.md)). Built against real weights and real
activations, the kernel measured **0.965x the bf16 two-pass control** once the
activation quantiser it needs is counted, against a 1.53x bar: the per-group
rescale costs more than the 2x instruction rate buys. It also showed **2.79%
relative L2 error** against the bf16 result, driven by activation outliers.
Details in [probe-w4a8-2026-09-23.md](probe-w4a8-2026-09-23.md).

So the same checkpoint serves both regimes and the activation path is the same
too. Symmetric checkpoints are still preferable - an asymmetric one forfeits the
mixed DPAS path outright, and would need a zero-point stream in every kernel -
but that preference no longer rests on a performance claim.

## Group size affects traffic, and nobody accounts for it

Scales are read alongside weights, so group size is not free:

| Scheme | Bits per weight (weights + fp16 scales) |
|--------|------------------------------------------|
| int4, g128 | 4.125 |
| int4, g64 | 4.25 |
| int4, g32 | 4.5 |
| MXFP4 (e8m0 per 32) | 4.25 |
| FP8 | ~8 |

g32 costs **9% more traffic than g128** for accuracy that may or may not matter
on a given model. Worth measuring per model rather than assuming smaller is
better. This project runs g64: finer scale granularity for about 3% more bytes
than the g128 default, and it matches a 64-element-K tile with inline f16
scales. Pick the group size your kernels' tile geometry wants.
