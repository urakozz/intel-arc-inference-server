# Spec 5 - int8 prefill linears: W4A16 checkpoint, i8 x i8 math

**Status:** implemented 2026-09-24 (plans 5a, 5b); `l0-int8` is the default (§8).

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

## 7. Amendment - 2026-09-24, results (plan 5b)

**Status after plan 5b: h8 is opt-in, the default stays `l0`. Gate A3 failed
while every 5a kernel test passes, so the stopping rule (§3) applies: the
numbers are recorded here and nothing was tuned.** A4 had not been run when
this was written (its tool-call set is plan 5c's).

### 7.1 What changed from the design

- **The switch is `--pp-backend l0-int8`**, not `--pp-math w4a16|h8`. The
  backend switch already reaches both CLIs, the engine and every test.
  `PrefillBackend::L0Int8` is L0 for attention, GDN, norms, the head and replay;
  only the int4 linears differ. Each adds one launch (the quantiser): +256 per
  chunk, measured equal to the prediction (9334 at C = 2048 with the head).
- **The column scales are built at the first l0-int8 prefill**, before its first
  chunk, not at load (§4 T2). That keeps recordings free of the host finish and
  the l0-only engine unchanged, and it puts about 140 ms into the first request
  (7.4).
- **A2 is redefined.** Prompt-logit relative L2 against the oracle needs an
  all-positions logits dump the engine does not have. A2 as gated: per golden
  prompt, over the 32 greedy decode steps after the prefill (each reads the KV
  and GDN state the prefill wrote), logit cosine against the CPU oracle with
  mean(int8) >= mean(l0) - 0.002 and min(int8) >= min(l0) - 0.01.
- **A3 as gated:** per prompt, the int8 gate verdict is PASS wherever l0's is,
  with the same determined-exact count (`tools/rotate/gate_compare.py`).
- **The `long` golden gate did not run.** `prefill_gate_long_test` and
  `prefill_gate_long_int8_test` are registered only while the RTN checkpoint and
  `oracle-out-long` exist, and neither exists on the box. Multi-chunk coverage
  stands in: a chunk-16 pair on the golden set (3 to 4 chunks per prompt).
  The chunk-1000 registration is single-chunk on these prompts (38 to 61 ids),
  so it repeats the default run's numbers exactly.

### 7.2 A2 and A3 (measured, `gate_compare.py`)

| run | prompt | l0 mean cos | l0 min | int8 mean | int8 min | l0 gate | int8 gate | A2 | A3 |
|---|---|---:|---:|---:|---:|---|---|---|---|
| chunk 0 (and 1000) | prose | 0.999971630 | 0.999909491 | 0.999949053 | 0.999795485 | 31/31, ties 1/1 | 31/31, ties 1/1 | PASS | PASS |
| | code | 0.999946584 | 0.999593069 | 0.999898762 | 0.999418297 | 32/32 | 32/32 | PASS | PASS |
| | cjk | 0.999961454 | 0.999839851 | 0.999947412 | 0.999785064 | 30/30, ties 2/2 | **29/30**, ties 2/2 | PASS | **FAIL** |
| chunk 16 | prose | 0.999971259 | 0.999930196 | 0.999958267 | 0.999908266 | 31/31 | 31/31 | PASS | PASS |
| | code | 0.999947747 | 0.999809837 | 0.999916840 | 0.999542875 | 32/32 | 32/32 | PASS | PASS |
| | cjk | 0.999964049 | 0.999919725 | 0.999951292 | 0.999910716 | 30/30 | 30/30 | PASS | PASS |

**A2 passes everywhere. A3 fails on cjk in the single-chunk run.** The failing
row is cjk greedy step 9, a determined row: the oracle's top two are
95895 at 13.9375 and 105874 at 13.875, one bf16 step apart; l0 picks 95895,
and h8 has 105874 at 13.96331 over 95895 at 13.95635, a 0.007 reversal. The
gate counts a determined row as not a judgement call, and so does A3 as
written. Split into 16-id chunks the same prompt passes 30/30, which says the
row sits inside h8's noise rather than beyond it, but the bar is the bar.

### 7.3 Engine smoke (measured, `prefill_int8_test`)

prose: same first id as l0, last-row logits cosine 0.99995. long[:2048]: same
first id, cosine 0.941. Over 30 prefixes of `long` the l0 / h8 last-row cosine
is 0.998 to 0.9999 on most and 0.88 to 0.95 on a scattered few, with the first
id equal on all 30: position-dependent sensitivity to h8's noise, not a shape
boundary (neighbouring lengths are good, the row is independent of padded
scratch rows and of a split at 256). Replay on l0-int8 is bitwise the
immediate run (logits, KV, GDN state, conv ring, Control).

### 7.4 B1 and B2 (measured, ITERATE grade: a CPU-only container was running, no GPU holder)

Interleaved on device 0: l0 1671.78 and 1668.74 t/s, l0-int8 1970.89 and
1971.09 t/s, paired ratios 1.1789 and 1.1812, **1.1798x the 1670.72 record:
B1 fails** (bar 1.20x, 2005 t/s). The bench row times a fresh process's first
prefill, which on l0-int8 includes the one-time scale pass: first call 2072 to
2076 ms, later calls 1929 to 1936 ms (+140 to +143 ms once; l0's first call is
29 to 30 ms faster than its later ones). A warm l0-int8 prefill is 2122 t/s,
1.27x (derived from host wall, not a bench row). Decode 29.40 t/s against
29.45: **B2 passes.** Numbers and runs in `docs/BENCHMARKS.md`, "The int8
prefill linears".

## 8. Amendment - 2026-09-24, the operator's ruling and the default

**A3's near-tie is accepted (operator, 2026-09-24).** The cjk row 9 flip is a
determined row whose oracle top-two margin is one bf16 ulp (0.0625 at 13.94),
reversed by h8 by 0.007. `prefill_gate_test` gains argv[7]: near-tie flips
accepted per prompt. A determined-row mismatch is accepted only if the engine
chose the oracle's runner-up **and** the oracle's own margin is at most one
bf16 ulp of its top logit, and it is printed as "near-tie ACCEPTED". The
allowance is 1 on `l0-int8` (explicitly on its registrations, and by default
when no argv[7] is given) and 0 on every other backend. `gate_compare.py`
counts accepted near-ties as exact. Re-run with the rule: every gate
registration passes, and `gate_compare.py` reports A2 PASS, A3 PASS for the
default pair (cjk `29+1nt/30`) and the chunk-16 pair.

**The scale pass moved to load, as §4 T2 said.** `Engine::prepare_prefill()`
does the lazy prefill setup and, on `l0-int8`, builds every int4 linear's
rotated column scales. `prefill()` calls it too (idempotent), and both CLIs call
it right after loading, so no request pays the ~140 ms. Decode-only engines
never call it, so ruling R7 stands.

**B1, re-measured with that change** (BENCHMARKS.md, "The int8 prefill
linears"): interleaved on an idle device 0, l0 1643.52 / 1639.56 t/s, l0-int8
2104.50 / 2102.64 t/s, paired **1.2805x / 1.2824x. B1 passes.** B2 stands at
29.40 t/s.

**A4, provisional.** The bf16 CPU baseline is incomplete: 15 of 36, and a
reboot interrupted it; it resumes where it stopped. Against W4A16 instead:
`l0-int8` makes a well-formed call on all 33 scenarios where `l0` does,
identical on 30. The other 3 are valid alternatives: grep against glob for
finding tests (twice), and one extra word in a rewritten comment. The other 3
scenarios truncate at 192 tokens on both backends. The formal A4 against
bf16 is recorded when the baseline completes.

**The default is now `l0-int8`** (`default_prefill_backend()` in both
`backend_sycl.cc` and `backend_sycl_absent.cc`). `--pp-backend l0` keeps the
bf16 walk, and every `l0` registration still runs it explicitly.

## 9. Amendment - gate A4, formal (2026-09-24)

The bf16 CPU baseline completed (36 of 36, `tools/toolcall/oracle_generate.py`,
greedy, 192 new tokens). `tools/toolcall/score.py toolcall-out bf16 l0 l0-int8`:

| run | first tool call matches bf16 (name and parameters) |
|---|---:|
| `l0` (W4A16 prefill) | 23 / 36 |
| `l0-int8` (h8 prefill) | **25 / 36** |

**A4 passes** (bar: `l0-int8` matches at least as often as `l0`). The same 3
edit scenarios truncate at 192 tokens on all three runs. Most mismatches are
shared by both engine paths, so they are the int4 checkpoint's differences from
bf16, not the int8 prefill's. The two scenarios where the engine paths differ
(t4_comment-linear_l0, t5_test-loader) both go to `l0-int8`. With A1 to A4, B1
and B2 all passed (A3 under the §8 ruling), spec 5 is complete.
