# Spec 5 - int8 prefill linears: W4A16 checkpoint, i8 x i8 math

**Status:** design, 2026-09-24, for operator review.

**Order, set by the operator:** this is the foundation. Flash attention,
prefix caching and MTP are built **on top of it**, after its gates pass, and not
before.

Every number is **measured** unless marked **derived** or **estimated**.

## 1. The decision this spec implements

The evidence is `docs/probe-w4a8-2026-09-23.md` sections 14 and 15. It rests
on three facts:

1. **The hardware.** The B70's int8 DPAS runs at 366.9 TIOP/s, 2.000x bf16. It
   has no FP8 or FP4 matrix path: undeclared builtins, compiler crashes, and the
   datasheet lists INT8 only (`docs/probe-dpas-rates-2026-09-22.md`).
2. **What is fast.** An i8 x i8 GEMM with **per-channel** weight scales and
   **per-token** activation scales, both applied in the epilogue only: 264 to
   297 TOP/s (§14.5). Checkpoint-group rescales inside the loop kill the rate
   (§11, §13).
3. **What is accurate.** Per-token int8 activations are accurate only after a
   1024-block Hadamard rotation (§14.2), and the rotation must be applied on
   **both** sides. End to end on four prompts, that path ("h8") matches W4A16
   within noise, while unrotated int8 costs a third to a half more error (§15.4).

So:

| | |
|---|---|
| **checkpoint** | `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`, **unchanged** |
| **decode** | unchanged: int4 GEMVs, bf16 activations |
| **prefill linears** | x' = x R_K per token, int8; W' = per-channel int8 of (W R_K), built per slab per chunk; i8 x i8 GEMM; y = acc x xs[m] x ws[n] |
| **R_K** | D_K * blockdiag(H_1024) / 32 over each linear's K (5120, 6144, 17408), fixed random signs per K |

What was ruled out, and why, is in §15.7: per-channel int4, offline rotation,
OpenVINO's IR, and the unrotated fast path as a default.

## 2. Where we start

The record (BENCHMARKS.md): prefill **1670.72 t/s**, 4096 ids as two 2048-id
chunks, 2451.6 ms. Decode 29.45 t/s.

Per 2048-id chunk, all 64 layers, probe timings at sustained clocks (§15.5,
**derived** by summing): the bf16 linears take **845.8 ms**. The same linears
on the h8 path take **602.5 ms**.

| fused linear (count) | control ms | h8 ms | h8 x |
|---|---:|---:|---:|
| gate‖up (64) | 6.083 | 4.442 | 1.369x |
| down (64) | 3.371 | 2.217 | 1.521x |
| GDN qkv‖z (48) | 2.811 | 2.091 | 1.346x |
| out_proj / o_proj (64) | 1.019 | 0.725 | 1.397x |
| FA q‖k‖v (16) | 2.539 | 1.848 | 1.375x |

Of h8's cost on gate‖up, the rotating requant is 1.32 ms, the GEMM 2.66 ms and
the rotating activation quantiser 0.056 ms. The requant is the one piece that
is not the matrix math.

## 3. Goal, bars, stopping rule

**Goal:** the engine's prefill runs every int4 linear on the h8 path,
selectable at runtime, at W4A16 accuracy.

**Accuracy bars.** These decide acceptance. Speed does not override them.

- **A1, kernels:** every new kernel is bit-exact against a CPU mirror of its
  arithmetic, as `probe_w8a8` already checks (requant, rotated requant,
  quantisers, GEMM oracle, slab against full).
- **A2, logits:** on the four §15.4 prompts plus the golden set (prose, code,
  cjk, long), the engine's prompt logits on h8 against the bf16 oracle fall
  within **+1.0 point** of the engine's W4A16 path's relative L2, and within
  **-0.01** of its mean top-5 overlap. §15.4 measured +0.5 points and -0.001
  to -0.010 in simulation.
- **A3, greedy agreement:** golden-set greedy continuations on h8 agree with the
  engine's W4A16 path at least as often as W4A16 agrees with the oracle today
  (doc 14's gate, `tests/golden`).
- **A4, tool calls:** a set of at least 30 agentic tool-call prompts (Qwen XML
  `<tool_call>` format, with tools rendered by the existing template). h8's
  greedy output must contain the **same tool name and the same parameters** as
  W4A16's on at least as many prompts as W4A16 matches bf16. Compared as raw
  text, so no tool-call parser is needed. Building this set is stage T0.

**Speed bar.**

- **B1:** pp4096 at least **1.20x** the record, so 2005 t/s or more (derived
  target from §2: 2451.6 ms minus 486 ms is about 1965 ms, 2084 t/s).
- **B2:** decode unchanged within noise (the path does not touch decode).

**Stopping rule.** If A2 to A4 fail on the engine while the kernels are
bit-exact, stop and record. The simulation was wrong about something, and more
kernel work will not fix it.

## 4. Stages, in order

### T0 - the tool-call acceptance set (before any engine code)

About 30 to 50 prompts taken from real agentic coding sessions: multi-turn
context, tools rendered, the next assistant turn being a tool call. Store ids
and expected-format notes under `tests/golden/toolcall/`. Record bf16 (oracle)
and W4A16 (engine) greedy outputs as the baseline. This is also the first
brick of the agentic-session benchmark the operator's workload needs later.

### T1 - production kernels, from the probe

Promote from `tools/probe/probe_w8a8.cl` to `src/kernels/prefill/`, unchanged
in arithmetic:

- `pw8_gemm` becomes `pf_gemm_i8`, with the epilogue variants production needs:
  - fp32 `partials` for the plain linears (`linear_l0`'s contract);
  - **SiLU x mul** writing bf16 `x` at pitch N/2 for gate‖up
    (`linear_l0_silu`'s contract, parity S2).
- `pw8_quant_had2_{5,6,17}` becomes `pf_quant_had`: bf16 x to int8 x' plus
  per-token scales.
- `pw8_requant_rot2` becomes `pf_requant_rot`: int4 g64 layout 0 to rotated
  per-channel int8, per 1024-column slab.
- **New: the layout-1 requant for GDN qkv‖z** (Interleave16). The probe timed
  its control with the layout-1 dequant but requantised layout 0.

Tests: A1, per kernel, in `tests/prefill/` beside `pf_gemm_test`.

### T2 - load-time column scales

ws[n] = max over k of |(W R_K)[k, n]| / 127 is a property of the weights. It
is computed once at load: one GPU pass per linear reusing `pf_requant_rot`'s
dequant and rotation, a max-reduction instead of a store. Stored as N fp32 per
linear, about 1.2 MB for the whole model (derived). The probe computed these on
the host.

### T3 - dispatch

`runtime::prefill::linear_l0` and `linear_l0_silu` gain an h8 branch: quantise
x (one launch per linear input, shared by the linears that read it), then per
slab `pf_requant_rot` and `pf_gemm_i8` on the same in-order list, with no host
wait, like today's slab walk. Selected by `--pp-math w4a16|h8`, default
**w4a16** until A2 to A4 pass. Scratch: one int8 slab ([K/4][1024] dwords,
17.8 MB at K = 17408), x' ([2048][K] int8, 35.7 MB at 17408) and the
per-token scales.

### T4 - gates, then the default

A1 (T1), then A2, A3, A4, B1 and B2 on the engine. Only if all pass does h8
become the default. The result is recorded in BENCHMARKS.md as a series row.

## 5. Out of scope, and on top later

- **Flash attention, prefix caching / session continuation, MTP speculative
  decoding:** after this spec, on top of it (operator's order).
- **Tool-call parsing** into OpenAI `tool_calls`. A4 compares raw text, and
  parsing is a server feature for later.
- **Fusing the activation quantiser into the preceding norm or SiLU kernel.**
  It is 0.056 ms per gate‖up, so it waits for a profile that says it matters.
- **Reducing the requant's cost** (1.32 ms of gate‖up's 4.44). rot3/rot4 did
  not beat rot2 (§14.6). Reusing a chunk's int8 slabs across chunks costs memory
  and is a later decision.
- **The ~0.7 s short-chunk floor** (§15.6). Real, and separate from this spec.

## 6. Measurement protocol

Kernels: interleaved control/candidate pairs after a warm-up, median paired
ratio with its range (§14.4). Sequential timing misstated ratios by up to 35 %.
Engine: `tools/bench_decode.sh --pp 4096`, median of three, idle box (DRM-holder
check), device 0. Accuracy: `tools/rotate/eval_quantised.py` and
`check_rotation.py` for the oracle side. The engine side needs a prompt-logits
dump in the same golden format. Whether the golden-gate harness already exposes
one is unchecked, so a CLI flag for it is part of T3 if not.
