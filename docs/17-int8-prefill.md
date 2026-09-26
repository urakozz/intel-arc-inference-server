# 17. The int8 prefill: how 2104.50 t/s is computed

The short version of spec 5 and the two weeks of probes behind it. The
evidence is `docs/probe-w4a8-2026-09-23.md` (sections 1 to 15), the design is
`docs/specs/2026-09-24-spec5-int8-prefill-linears-design.md`, and the rows are
in `BENCHMARKS.md`. Every number is **measured** unless marked **derived**.

## What runs

```
stored once (the checkpoint)      int4 weights, one scale per 64 (g64)
                                        │
DECODE (unchanged)                 int4 → bf16 on the fly, bf16 math        ~29.4 t/s
                                        │
PREFILL, per linear layer, per chunk, on the fly:
  activations  x (bf16)  ──► rotate (Hadamard) ──► int8, one scale per token
  weights      int4 g64  ──► dequant ──► rotate (same Hadamard) ──► int8, one scale per column
                                        │
                           int8 × int8 on the XMX engines (2x bf16 rate)
                                        │
                           rescale → bf16/fp32 output (the rotation cancels exactly)
```

- **Nothing new was quantised.** The file is the original
  `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`. W4A16 is what is *stored*;
  W8A8 is what prefill *computes*, rebuilt from it on the fly for every
  1024-column slab of every chunk.
- **The rotation** is R = D x blockdiag(H_1024) / 32 (random signs, a
  Walsh-Hadamard per 1024 channels). It spreads the activations' outlier
  channels over all channels, so int8 with one scale per token loses little.
  Applied to both sides, x R and R^T W, it cancels: (x R)(R^T W) = x W.
- **The weight scales** (one per output column of the rotated weights) are a
  property of the weights. They are computed once, when the model loads.
- **`--pp-backend l0-int8`** is the default. `--pp-backend l0` keeps the bf16
  prefill.

**Result:** pp4096 **2104.50 t/s**, against 1643.52 for the bf16 prefill in the
same session, paired **1.28x**. Decode is unchanged at 29.40 t/s. Accuracy on
the golden gate matches the bf16 prefill, with one accepted bf16 near-tie
(spec 5 §8).

## Why decode stays int4 and bf16

Decode multiplies one token's row by every weight matrix. With one row there is
no matrix for XMX to work on, and the multiply is not the bottleneck: reading
the weights from memory is. Decode speed is therefore set by bytes per weight:

| stored weights | bytes per weight | decode (derived) |
|---|---:|---|
| int4 g64 (today) | ~0.53 | 29.4 t/s measured |
| int8 or FP8 (a W8A8 / FP8 checkpoint) | ~1.0 | roughly half, mid-teens |

An 8-bit checkpoint would make decode **slower**, not faster, and FP8 has no
matrix hardware on the B70 anyway (only INT8 in Intel's datasheet; the FP8
builtins are undeclared or crash the compiler). Decode gets faster from fewer
bytes, or from more tokens per weight read: MTP speculative decoding or batching,
which turn decode into a small GEMM that XMX can run.

## How we got here, and what was rejected

| # | tried | result | verdict | record |
|---|---|---|---|---|
| 1 | FP8 / FP4 matrix math | not on this silicon: undeclared builtins, compiler crashes, INT8-only datasheet | **impossible** | probe-dpas-rates |
| 2 | int4 weights x int8 activations at the checkpoint's g64 (i8 x i4 DPAS) | 0.97x the bf16 control: the per-group rescale every 2 instructions eats the 2x | rejected, speed | probe-w4a8 §1-12 |
| 3 | the same at coarser groups | g128 1.22x, g256 1.43x, per-channel 1.75x (speed only) | only per-channel is fast | §13 |
| 4 | int8 activations without rotation (CPU, real layers) | gate‖up 2.8 %, down 5.1 % error, worst rows 0.998 / 0.957 | rejected, accuracy | §14.2 |
| 5 | SmoothQuant instead of rotation | helps gate‖up, fails on down (outliers move per token) | rejected | §14.2 |
| 6 | int4 activations (W4A4) | 16x coarser than int8, which already failed unrotated | not pursued | §8 |
| 7 | rotating the weights inside the engine: rot1 to rot4, including an exact DPAS factorisation | rot2 (SLM butterflies) fastest; the DPAS form was exact but not faster | **rot2 kept** | §14.6 |
| 8 | a rotated checkpoint, made offline (rotate bf16, then AutoRound) | the rotation is exact (3.9e-7) but makes int4 weights ~25 % worse | rejected; model deleted | §15.1-15.2 |
| 9 | per-channel int4, AutoRound on the rotated model | 22.9 % logit error vs 7.77 % for g64 | rejected; checkpoint deleted | §15.2 |
| 10 | OpenVINO's own int4 IR (asym g128, dynamic int8 activations) | 1298 t/s prefill, 21.6 decode, 26 % logit error | rejected | §15.3 |
| 11 | the unrotated fast path (int8 activations and weights, no Hadamard) | 1.70x / 1.85x on gate‖up / down, but 17.8 % vs 12.0 % end-to-end error | rejected as default | §15.4 |
| 12 | an 8-bit (W8A8 / FP8) checkpoint | would halve decode and take ~27 GB of the 32 | not pursued (derived) | above |
| 13 | **runtime Hadamard + per-channel int8, W4A16 file unchanged ("h8")** | 12.5 % vs 12.0 % end-to-end on four prompts; 1.35x to 1.52x per linear; 1.28x whole prefill | **accepted, the default** | §15.4-15.5, spec 5 |

What decided it: rows 4 and 5 said the activations need a rotation, row 11 said
the rotation cannot be skipped, rows 8 and 9 said it cannot be moved into the
file, and row 7 made it cheap enough to do at runtime.

## What the speed is made of

Per fused linear, 2048-token chunk, paired against the bf16 slab walk
(probe_w8a8, §15.5):

| linear (per model) | bf16 ms | h8 | of which: rotating requant, GEMM (gate‖up) |
|---|---:|---:|---|
| gate‖up (64) | 6.083 | 1.369x | 1.32 ms, 2.66 ms of 4.44 |
| down (64) | 3.371 | 1.521x | |
| GDN qkv‖z (48) | 2.811 | 1.346x | |
| out_proj / o_proj (64) | 1.019 | 1.397x | |
| FA q‖k‖v (16) | 2.539 | 1.375x | |

The int8 GEMM itself runs at 264 to 297 TOP/s, 2.0x to 2.3x the bf16 GEMM.
Part of that 2x is spent rebuilding the rotated int8 weights every chunk: the
requant is the one cost here that is not matrix math.

## Open, and closed

- (closed) **Gate A4, tool calls**, 36 agentic prompts against the unquantised
  bf16 model: `l0-int8` matches **25 / 36**, the W4A16 prefill `l0` 23 / 36
  (spec 5 §9).
- **The short-prompt floor:** about 0.7 s of prefill whatever the length
  (probe-w4a8 §15.6). It is separate from int8 and matters for every
  agentic turn once prefix caching lands.
- (done) **Flash attention and 128k context**, spec 6, on top of this path:
  pp4096 2125.12 t/s (docs/BENCHMARKS.md, "Flash attention and 128k").
- **Next:** prefix caching, MTP.
