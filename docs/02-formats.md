# Format decision

**Decision: int4 weight-only, 16-bit activations (W4A16), normalised to one
canonical on-device layout at load time.**

## Why not the alternatives

| Candidate | Verdict |
|-----------|---------|
| **FP8** | Rejected. 8 bits/weight = 2× int4 traffic on a decode that is bandwidth-bound, *and* Xe2 has no native FP8 XMX so it upconverts anyway (`MainloopIntelXeXMX16FP8Scaling`). Its entire advantage elsewhere is hardware that this card lacks. |
| **OpenVINO IR** | Rejected as a runtime format. It is a graph-and-weights container tied to the OpenVINO runtime; adopting it means adopting their execution model, which is the thing this project exists to replace. Worth reading `~/PycharmProjects/openvino.genai` for its scheduling ideas, not for its file format. |
| **MXFP4** | Deferred to phase 3. Same hardware path as int4, but block-scaled support in `sycl-tla` is `xe35`-only, so it is not free on Battlemage. |
| **int4 GPTQ** | Supported - same tensor layout as AutoRound (see below). |
| **int4 AutoRound** | **Primary.** Best accuracy at 4 bits of the options we have checkpoints for. |

## AutoRound vs GPTQ: a config difference, not a layout difference

Both 9B checkpoints on the box declare:

```
quant_method    "auto-round"
packing_format  "auto_round:auto_gptq"
bits            4
sym             true
group_size      128   (letechlead)  /  64  (urakozz)
```

The packed nibble layout **is** GPTQ's. What differs:

- the **offline rounding algorithm** (AutoRound's signed-gradient search vs GPTQ's
  Hessian-based rounding) - affects weight values, not their arrangement;
- the **metadata**: AutoRound adds `extra_config` / `dynamic` per-layer overrides
  that let a single checkpoint mix bit-widths.

So **one unpacking routine covers both formats.** Only the metadata parser
differs - and it must accept two spellings of the same thing. The phase-1
checkpoint (`Vishva007/Qwen3.8-27B`, read on the box) says
`quant_method: "gptq"`, `provider: "auto-round"`, `autoround_version: 0.15.0`,
`desc_act: false`, `group_size: 64`, `sym: true`, `lm_head: false`, while the
9B says `quant_method: "auto-round"`, `packing_format: "auto_round:auto_gptq"`.
Same nibbles on disk. The loader keys on the tensor suffixes
(`qweight`/`scales`/`qzeros`/`g_idx`), not on the label.

What GPTQ packing actually ships per linear, from the headers
(`in_proj_qkv`, K=5120, N=10240, g64):

| Tensor | dtype / shape | Bytes | Read at decode? |
|---|---|---|---|
| `qweight` | I32 `[K/8, N]` = `[640, 10240]` | 26.2 MB | yes - the traffic |
| `scales` | F16 `[K/64, N]` = `[80, 10240]` | 1.6 MB | yes - 6.25% on top |
| `qzeros` | I32 `[K/64, N/8]` = `[80, 1280]` | 0.4 MB | **no** - `sym: true` makes every zero-point the constant 8; drop at load |
| `g_idx` | I32 `[K]` | 20 KB | **no** - `desc_act: false` makes it the identity; assert and drop |

Across the whole model `qzeros` + `g_idx` are 0.20 GB of 15.7 GB - a free 1.3%
that a kernel reading the GPTQ layout verbatim pays every token. AutoRound also ships an AWQ-packed variant (`auto_round:auto_awq`) whose
nibble order is `[0,2,4,6,1,3,5,7]`; converting it to GPTQ order is a lossless
permutation, and is a load-time concern.

### Is AutoRound still needed?

**Yes - the algorithm. No - the runtime.**

AutoRound is an offline quantiser that produces better int4 weights than
round-to-nearest. Keep using it (or consume checkpoints others produced with it).
What this project drops is **ARK** (`auto_round_kernel`), its runtime
`woqgemm` library. We write that ourselves.

Clean separation: *keep the recipe, discard the oven.*

## Mixed bit-width is a real hazard

AutoRound's `dynamic` rules let one checkpoint carry layers at different widths.
A 27B checkpoint on the box (`Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound`) has
**97 layers at bf16, 17 at int8, the rest int4** - the int8 ones being attention
`out_proj`/`o_proj`, two `down_proj`, and layer 0's `in_proj_qkv`/`in_proj_z`.

The loader must therefore treat **per-layer width as data, not as a global
constant**, from day one. Retrofitting that is painful; vLLM's XPU path still
hard-gates it and raises `NotImplementedError` on 8-bit.

The phase-1 checkpoint already exercises this in a mild form: its 98 `dynamic`
exclusion rules leave `linear_attn.in_proj_a` / `in_proj_b` in **bf16** in all
48 GDN layers (`[48, 5120]` each, 0.5 MB - the gate/decay projections), plus
the whole MTP head in bf16 (0.85 GB). Everything else is int4. The loader
reads the exclusion list and the tensor suffixes; nothing is assumed.

## Design principle: format plurality is a load-time problem

All supported formats normalise into **one canonical on-device layout** during
model load. Consequences:

- **Zero runtime branching.** The decode kernel knows one layout. No dispatch on
  `packing_format`, no `is_awq_packed` checks in the hot path.
- **Pre-swizzling is free.** Rearrange weights once at load into whatever order
  the XMX mainloop wants. Every CUDA backend does this; vLLM's XPU MXFP4 path
  notably does not (`compressed_tensors_moe_w4a4_mxfp4.py:190` is a bare `pass`).
  This is measurable headroom sitting on the floor.
- **Adding a format is a loader change**, not a kernel change.

The canonical layout is chosen by what the `sycl-tla` mixed-dtype mainloop wants,
not by what any checkpoint happens to ship.

**Canonical int4 layout chosen 2026-08-23 by `probe_gemv`: layout 1** (the tiled
544-byte block per 16 `n` × 64 `k`, scales inline) - 2712 vs 2666 GB/s summed
over the five decode shapes, so **the two layouts are within 2% of each other**
by the deciding metric (3.7% apart in wall time). The whole lead comes from one
shape, `qkv‖z` at N = 16384, where layout 0's power-of-two 64 KB row stride caps
it at 420 GB/s against layout 1's 560; on the other four shapes layout 0 is
0.9-6.3% ahead. Recorded as the rule dictates, and worth re-running if the shape
mix changes - the full table and the argument are in
[12-kernels.md](12-kernels.md).

## W4A16 for decode, possibly W4A8 for prefill

The *weight* format is settled (int4, symmetric). The **activation** width is a
separate decision, and it should differ by regime:

| | Activations | Why |
|---|---|---|
| Decode | bf16 (W4A16) | Bandwidth-bound. The multiply is not the cost and activations are one vector - quantising them buys nothing. |
| Prefill | **int8 (W4A8)** - candidate | Compute-bound. Unlocks Battlemage's **native mixed s8 × s4 DPAS**, skipping int4→16-bit upconversion entirely (see [01-hardware.md](01-hardware.md)). |

The same checkpoint serves both; only the activation path differs.

The s8×s4 DPAS requires **no zero points on the int4 matrix**, which is why
`sym: true` matters beyond accuracy - an asymmetric checkpoint forfeits this path
outright. Prefer symmetric checkpoints for that reason alone.

Unproven - open question 9.

## Group size affects traffic, and nobody accounts for it

Scales are read alongside weights, so group size is not free:

| Scheme | Bits per weight (weights + fp16 scales) |
|--------|------------------------------------------|
| int4, g128 | 4.125 |
| int4, g64 | 4.25 |
| int4, g32 | 4.5 |
| MXFP4 (e8m0 per 32) | 4.25 |
| FP8 | ~8 |

g32 costs **9% more traffic than g128** for accuracy that may or may not matter on
a given model. Worth measuring per model rather than assuming smaller is better.
