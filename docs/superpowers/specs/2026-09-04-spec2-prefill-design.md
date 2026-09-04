# Spec 2 - Prefill

Status: design for review, 2026-09-04. Follows spec 1.7 (closed on the
operator's 32.0 t/s line; tag `spec1.7-done`). Phase 1's target has two
halves - `tg256 > 31.50` and `pp4096 ≥ 1973` - and only the first is met.
This spec is the second half: a real prefill for the same model, on the same
engine. Tokenizer and HTTP are **spec 3** (operator ruling, 2026-09-04).

Every number below is measured unless marked **derived** or **estimated /
external**; the source is cited. The five explorer reports that ground it
(engine readiness, prefill roofline, DPAS/prior art, tokenizer+HTTP scope,
vLLM prefill baseline - 2026-09-04) are summarised in §1 and are filed in
the spec-2 SDD workspace when the plan opens.

## 1. Where we start (all measured or read from source, 2026-09-04)

**The engine has no prefill.** `Engine::ingest()` replays the decode list
once per prompt id (`src/runtime/engine.cc:77-83`); a 4096-token prompt
costs **121 s** (bench log, `2a7df0b`, RTN checkpoint) against vLLM's
**~2.1 s** on the same box. The `M` dimension that already exists is not a
prefill dimension: it is capped at 8 by `Control` and at 13 by the GDN conv
ring, it was built for MTP, and `gdn_step` is a genuine sequential
recurrence over positions (state decay + rank-1 update carried in registers
across the `m` loop, `gdn_step.cl:283-381`). Raising `M` makes it correct,
not parallel. `attn_part`, the decode attention scratch, scales linearly in
`M` (50.7 MB at M=1 → ~13 GB at M=256, `buffers.h:84`).

**The physics.** Per token the forward pass is **48.97 GFLOP (derived from
`docs/03-models.md` shapes)**, 70% of it in the MLP (`intermediate_size`
17408). Weights are read once per chunk; FLOPs scale with chunk width. The
bandwidth/compute crossover on the measured 590 GB/s is at **~11 tokens on
vector fp32, ~87 on XMX bf16 (derived)** - every useful chunk is
compute-bound. vLLM's 1973 t/s implies **~98 TFLOP/s sustained (derived)**,
which is **2.1× the optimistic vector-only ceiling (45.9 TFLOPS, estimated)**.
Conclusion, and it is the design's foundation: **prefill cannot be reached
by widening the GEMV kernels; it requires the XMX systolic path.**

**The XMX path is reachable from our toolchain.** Verified on the box
(`ocloc 26.27`, `bmg-g31`): `intel_sub_group_bf16_bf16_matrix_mad_k16`
(and the f16 / i8 / u8 / tf32 forms) compile to a real `dpas.8x8` with
`has_dpas: true` in the zebin. M ∈ {1,2,4,8}, K = 16 (bf16), N = 16, fp32
accumulate. The **int4 named builtins declare but do not lower** to a `dpas`
on this driver; the instruction itself (`dpas ... :s4`) is real and reachable
only via inline vISA. No SYCL is *required* - but see §3 for why we use it
anyway.

**What Intel already has, and what it lacks.** `sycl-tla`'s BMG GEMM
(256×256×32 work-group tile, 32 subgroups, no SLM, 2D block loads, 2-stage
prefetch) is the same *family* as the mainloop vLLM's W4A16 prefill runs -
**not the same mainloop**: vLLM takes a mixed-precision variant
(`MainloopIntelXeXMX16MixedPrecision`, int4 upconverted in-register), while
the stock bf16 GEMM at our pin is `MainloopXeL1Staged<2>` (plan 6a, read
from the source at the pinned sha). We take the stock bf16 one because the
dequant scratch (§3.1) does the upconversion. Its FMHA forward
has varlen and paged-KV support; the shipped example is head_dim 128 -
ours is **256**. **There is no chunked GDN kernel in sycl-tla**; vLLM on XPU
falls through to the Triton reference (`fla_chunk_gated_delta_rule`,
`FLA_CHUNK_SIZE = 64`) for 48 of our 64 layers. An Intel CuTe kernel for it
exists in `vllm-xpu-kernels` (`chunk_gated_delta_rule_kernels_xe2.hpp`,
1634 lines) and is not on vLLM's traced path.

**The bar's anatomy.** vLLM's `pp4096` is **two chunked steps of 2048**
(`max_num_batched_tokens` defaults to 2048 on a 32 GB card; the V1
scheduler enforces it even for a solo request). `pp` as llama-benchy
measures it includes HTTP receipt, tokenization, scheduling and first-token
sampling (`est_ppt = ttfr − probe latency`). A device-side number from this
spec is therefore a **narrower quantity** and is labelled as such until
spec 3's server exists.

**An addition to our own record (not a correction - this spec's first
version said otherwise and was wrong).** `docs/06-prior-art.md:59` already
names oneDNN's JIT GEMM generator `gemmstone` and calls it "the most
valuable checkout after `sycl-tla`"; there is no "closed" claim anywhere in
that file (verified by grep, 2026-09-04). What is genuinely new is the
**call chain**: vLLM's XPU int4 prefill goes
`int4_gemm_w4a16` → `dnnl_matmul_w4a16_int4` → oneDNN → `gemmstone`, whose
cost-model selector is keyed on `m` and therefore picks a different
microkernel for prefill's M than for decode's M=1. §9 records that
additively.

## 2. Goals, bar, stopping rule

**Goals (operator, 2026-09-04):** a custom engine that is **faster than
vLLM** and usable for real serving. Learning the hardware is incidental,
not a goal: **wherever Intel's SYCL code already solves a term, it is
inherited; low-level code is written only where nothing exists.** The work
is divided into independent streams with fixed interfaces so it can be
delegated and run in parallel (§3.7).

**The bar is set from measurement, not guessed** (operator ruling). Stage 0
(§4) measures every term of a prefill step at the production shapes; the
**composed ceiling** is their sum at the chosen chunk width. The gate bar is
**a margin of that ceiling, ruled at Stage 0 close** - recommended default
**90%**, the same discipline spec 1.7 used to price levers. Two absolute
statements hold regardless of the margin:

- the row is quoted beside vLLM's **1973 t/s** with both labels attached
  (device-side vs HTTP-inclusive; chunk width; checkpoint);
- if the composed ceiling itself lands under vLLM, the memo says so - that
  is the "skill or silicon" answer for prefill and it is a result, not a
  failure.

**Stopping rule:** the ladder in §5 runs in order; each stage has its own
correctness gate; when the ladder is exhausted the gate (§6) runs; short →
re-assessment memo to the operator; nothing further starts without a
ruling.

## 3. The design - D: decouple weights, inherit everything Intel ships, own only the gaps

### 3.1 The one coupling, and how it is cut

Decode reads weights in per-shape tile layouts that Task 4 just tuned
(−1.20 ms/token). `sycl-tla`'s mixed-precision mainloop reads int4 in *its*
layout. A second weight copy costs ~12 GB; retuning decode's layout to suit
prefill sacrifices a measured win. **The cut: a bf16 dequant scratch.**

- **`dequant_tile` (ours, OpenCL C):** reads our layout-0/1 tiles with the
  dequant contract `gemv.cl` already implements (`w = (q − 8)·scale`, inline
  f16 scales, xor-shift form), writes row-major bf16 into **one reusable
  scratch** sized for the largest matrix (gate‖up: 5120 × 34816 × 2 B =
  **356 MB**). Bandwidth-bound: ~1.2 ms for that matrix at 590 GB/s
  (derived); Stage 0 measures it.
- Intel's own sweep on BMG found exactly this path the winner: their fused
  int4-DPAS variants regressed "*below the legacy bf16-dequant + stock GEMM
  fallback for every shape*" (`auto_round_extension/ark/.../sycl_tla_moe_prefill_s4_dpas.hpp:11-18`),
  because they materialised an upcast *workspace* per GEMM; a single
  reusable scratch per matrix is that same fallback, with the bandwidth
  overhead priced in Stage 0 (**derived: 8-16% of chunk time at C =
  4096-2048**; single-stream can take the wider chunk).
- Decode's layouts, kernels and capture are **untouched** by this spec.

### 3.2 The plain-GEMM interface - inherited

`prefill_gemm(A: bf16[M×K] row-major, B: bf16[K×N] scratch, C: fp32|bf16[M×N], M runtime, K, N)`.

One implementation: **`sycl-tla`'s stock bf16 GEMM**. Intel's tuned tile,
built with `icpx -fsycl-targets=spir64_gen -device bmg-g31`
(`/opt/intel/oneapi/compiler/2026.1/bin/icpx`, `docs/10-the-box.md:42`),
launched on a SYCL in-order queue **constructed from our existing Level
Zero context/device handles** (`sycl::ext::oneapi::level_zero` interop),
operating on the USM device pointers the engine already owns. Runtime `M`
handles the prompt tail without per-M binaries.

The interface exists so the GEMM is testable and swappable, not because a
second implementation is planned. The verified fact that our OpenCL C +
`ocloc` toolchain can emit `dpas` (§1) is recorded as the **fallback** if
`sycl-tla` ever has to be replaced - it is not a work item in this spec.

Why inherit rather than write, stated plainly: on equal bytes our decode
kernels are still 5.7% behind vLLM's after three specs in the regime we
know best (docs/BENCHMARKS, spec-1.7 gate rows). Out-tiling Intel's GEMM
on a first attempt in a regime we have never touched is optimism, not
evidence - and it is not the goal.

### 3.3 Prefill attention (16 full-attention layers)

Causal attention of a chunk of `C` new positions over the cache `[0, pos+C)`.
head_dim 256, GQA 6:1 (24 q / 4 kv), partial RoPE on 64 of 256 dims, the
`attn_output_gate` applied after (all as `attn_prep`/`prep_gated_head` do
today). Flash-style streaming is **mandatory from the smallest chunk** - a
naive S×S bf16 score block at S = 256 is exactly the 128 KB SLM budget for
one head (derived, `docs/01-hardware.md:16`).

- The chunk's K/V are written into the **existing cache layout** by a
  widened `attn_prep` (it is already M-parallel and indexes by `pos+m`),
  so decode continues from the same cache with no relayout.
- **`sycl-tla` FMHA forward first, if it instantiates at head_dim 256** -
  Stage 0 checks. It reads K/V from our cache (page size = whole cache,
  cumulative-seqlen arrays of one entry) and handles causal masking by
  tile trimming + diagonal masking.
- **Fallback, ours, only if P4 fails:** a flash kernel - Q-tile per
  work-group, online softmax, DPAS for both QKᵀ and PV - written in SYCL
  on top of `sycl-tla`'s own copy/MMA atoms if that is the shorter path
  (head_dim is a template parameter there), OpenCL C otherwise, with a
  pre-registered price. Task 5's register-packed GQA (one K/V walk serves
  six q-heads) transfers directly.

Attention is **~3% of prefill FLOPs at depth 4096 (derived)**; its risk is
memory tiling and correctness, not throughput.

### 3.4 GDN chunked delta rule (48 linear-attention layers) - inherit first, own if it does not fit

The **WY-representation chunked gated delta rule** (Yang et al.; the `fla`
reference vLLM runs): intra-chunk size **64**, a triangular solve makes the
64 positions' rank-1 updates parallel, and only `C/64` sequential steps
carry the 128×128 fp32 state between chunks. It must read and write **the
same `gdn_state` and `conv_ring` buffers `gdn_step` uses**, so decode
continues without translation.

**First candidate: Intel's CuTe kernel** `vllm-xpu-kernels/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp`
(1634 lines, Xe2, not on vLLM's traced path). With SYCL in the build it is
usable, not just readable. Stage 0 (P5) checks three things: it builds
against our toolchain; its state/conv layout matches ours or can be
adapted by a cheap relayout at chunk boundaries; its fp32 numerics keep
the §6 gates. If all three hold it is the kernel. **If not, ours:**

- **Batched conv1d** (depthwise 4-tap causal) over the chunk, seeded from
  the ring's last 3 positions, writing the chunk's last 3 back - the ring's
  `M + 3 ≤ RING` argument (`gdn_step.cl:176-178`) is replaced by explicit
  seed/writeback, not by a deeper ring.
- **The intra-chunk block:** cumulative gate sums, `A = (K·Kᵀ ⊙ β)`,
  `solve_tril`, `W, U` recompute, then the chunk-to-chunk state scan. GEMM-
  shaped at 64×128×128 per head; goes through the §3.2 interface where the
  shape suits DPAS, plain vector code where it does not (the solve).
- **FLOPs are negligible** (recurrence 5.5 MFLOP/token vs 772 M for the
  layer's GEMMs, derived). This kernel is a **correctness** problem: it
  replaces a bit-exact sequential recurrence with an algebraically equal,
  differently-rounded one across 48 layers. §6's bars are written for that.

### 3.5 Everything else in the chunk - widened, ours, OpenCL C

RMSNorm (`prep_res_fold`/`prep_norm_finish`), SiLU·mul, the a/b
projections, embed gather: all already M-parallel per kernel; they get
runtime-`M` variants (grid over `ceil(M/tile)`, masked), compiled AOT as
today. `lm_head` runs **only on the chunk's last position** through the
existing `LmHead` S=1 route; the sampler writes `out_token`/`cur_token` and
advances `pos` exactly as `argmax_stage2` does, so the **handoff to decode
is the same `Control` protocol** (`src/runtime/control.h`): after prefill of
`L` ids, `pos = L`, the cache/state hold positions `[0, L)`, and
`cur_token` is the first generated id - indistinguishable from what
ingest-by-decode would have produced, up to rounding.

### 3.6 Execution model

- **Decode:** unchanged - one captured L0 list, replayed per token.
- **Prefill:** **dynamic dispatch**, one in-order execution context. Our
  OpenCL C kernels launch with runtime arguments on an L0 immediate
  command list; `sycl-tla` kernels on the SYCL queue built from the same
  context; ordering by events (or, first cut, a queue wait between the
  two). Launch overhead is irrelevant at millisecond kernel scale
  (`docs/04-architecture.md:150-178` accepted this from day one). Stage 0
  measures the interop cost and requires it under 1% of chunk time.
- **Chunk width `C`:** single-stream, so **the widest that fits** -
  default target **4096** (one pass for the benchmark prompt; dequant
  overhead halves vs 2048); fallback 2048. Memory at C=4096 (derived):
  weights 13.7 GB + KV 1.07 GB (16k) + dequant scratch 0.36 GB +
  activations ≤ 0.5 GB - fits. Prompts longer than `C` run in chunks; the
  attention/GDN paths are written for multi-chunk from the start (§6 tests
  it).
- **Buffers:** `DecodeBuffers` splits into a **persistent group**
  (`control`, `gdn_state`, `conv_ring`, `kv_k`, `kv_v` - shared) and
  per-path scratch (decode's at `kM`, prefill's at `C`). The explorer
  confirmed two `CapturedStep`s over one buffer set is already how
  `--profile` works (`capture.cc:143-151`).

### 3.7 Divide and conquer - four independent streams, fixed interfaces

The design is cut so each stream can be delegated and built in parallel
against an interface the others can stub:

| stream | owns | interface it exposes | can be tested without |
|---|---|---|---|
| **S1 - SYCL build + GEMM** | `sycl-tla` as a pinned static dependency; the L0↔SYCL interop; `prefill_gemm`; `dequant_tile` | `prefill_gemm(A,B,C,M,K,N)` on USM pointers; `dequant_tile(shape) → scratch` | the engine: a standalone harness times it at the production shapes |
| **S2 - GDN chunk** | the chunked delta rule (Intel's or ours) + batched conv1d | `gdn_prefill(layer, x[C], state, ring) → y[C]`, same buffers as `gdn_step` | everything else: a CPU-torch reference and the decode `gdn_step` as the self-consistency oracle |
| **S3 - attention** | widened `attn_prep` (cache write) + FMHA or our flash kernel | `attn_prefill(layer, q[C], cache, pos) → o[C]` | S1/S2: a synthetic-cache harness like `probe_attn` |
| **S4 - plumbing** | buffer split, runtime-`M` variants of the M-parallel kernels, `Engine::prefill()`, the `Control` handoff, `b70-decode --pp`, the gates | the prefill step that calls S1-S3 | real S1-S3: runs on widened GEMV + `C ≤ 64` decode attention until they land |

S4 is the integration stream and the only one with ordering
dependencies; S1-S3 are independent of each other and of S4's early
stages. The plan assigns one implementer per stream.

## 4. Stage 0 - probes before levers (pre-registered predictions mandatory)

All at the five production prefill shapes {QkvZ 5120×16384, OutProj
6144×5120, GateUp 5120×34816, Down 17408×5120, Qkv 5120×14336} plus
`lm_head`, M ∈ {512, 1024, 2048, 4096}, card 1 while a server occupies card
0, warm-up discipline per `gemv_harness.h`.

| probe | measures | decides |
|---|---|---|
| **P1 - interop smoke** | SYCL queue from our L0 handles; USM pointer round-trip; event ordering; per-launch overhead | the execution model is real before anything is built on it |
| **P2 - `sycl-tla` GEMM** | TFLOP/s per shape and M; determinism (no split-K atomics in the chosen config); **prediction: ≥ 90 TFLOP/s at M = 4096 (vLLM's implied 98 is the yardstick)** | the production GEMM's ceiling; the first measured XMX number this project has |
| **P3 - dequant scratch** | `dequant_tile` GB/s and ms per matrix; **prediction: ≥ 500 GB/s** | the 8-16% overhead claim; whether scratch reuse across layers needs double-buffering |
| **P4 - FMHA at head_dim 256** | does it instantiate; µs at C ∈ {1024, 2048, 4096} over depth 4096 | inherit vs own attention |
| **P5 - Intel's CuTe GDN kernel** | builds; layout fit vs our `gdn_state`/`conv_ring`; correctness vs `torch_recurrent_gated_delta_rule` at fp32; state max-rel-diff after 4096 positions; µs per layer per chunk | inherit vs own GDN; the numerics band §6 must tolerate |

**Stage 0 output:** a `docs/12` prefill probe matrix; the **composed
ceiling** at the chosen `C` (sum of P2-P5 terms + norms/SiLU/embed priced
from their decode rows × M); the **90%-margin bar proposal**; the
operator's ruling on the bar. Also the first **measured** XMX throughput
number for `docs/01-hardware.md`, replacing the derived 183.5 TFLOPS.
Streams S1-S3 each own their probe; Stage 0 is itself parallel.

## 5. Stage 1 - the ladder

**L1 - Plumbing + correctness on widened GEMV (no XMX yet).** Buffer split,
runtime-`M` variants of the M-parallel kernels, batched conv1d + the
chunked GDN (§3.4), widened `attn_prep`, a *temporary* prefill attention
over the existing `attn_decode` at small `C` (its scratch is linear in `M`,
so `C ≤ 64` here), the `Control` handoff, `Engine::prefill()`, and
`b70-decode --pp`. **Gate:** §6 correctness bars on the short golden
prompts. Nothing about speed; this stage exists to put the hardest
correctness problem on the table first. Yield: prefill becomes *possible*;
throughput is whatever widened GEMV gives (estimated 5-10× today).

**L2 - The GEMM swap.** `dequant_tile` + `sycl-tla` GEMM behind the §3.2
interface at `C` = 4096 (or 2048 by P2/P3). **Gate:** correctness bars
unchanged (the interface is numerically a bf16 GEMM either way); the
long-prompt multi-chunk gate (§6) added. This is where the order-of-
magnitude lands.

**L3 - Attention.** FMHA (P4) or our flash kernel; retires the `C ≤ 64`
limitation. **Gate:** correctness bars + attention-family attribution.

L1 is S4's integration milestone; L2 and L3 are S1's and S3's landings.
S2 lands inside L1. There is no "learning lever": once the ladder is
exhausted the gate runs.

Per-stage acceptance: full suite green; the §6 bars on that stage's
checkpoint; `--profile`-style per-kernel attribution of the prefill step;
docs/12 mechanism in the same commit; miss → revert + priced record.

## 6. Correctness bars (hard)

1. **Golden gate after prefill, tie-aware, unchanged semantics.** For each
   golden prompt (prose/code/cjk): `prefill(prompt)` then generate → the
   continuation must satisfy the existing gate against the CPU-torch
   oracle (determined rows exact; tie rows set-membership; teacher-forced
   tails). Run on **both** the int4-`lm_head` (RTN) and Vishva
   checkpoints.
2. **Multi-chunk golden gate.** A new oracle prompt ≥ 2048 ids (one CPU
   oracle run, hours-class - estimated from the 18-minute 32-id runs -
   recorded in `oracle-out-long`), gated at
   `C = 1024` so every chunk boundary is crossed in attention-over-cache
   and in the GDN state carry.
3. **Self-consistency control.** `prefill(P)` vs `ingest-by-decode(P)`:
   64 generated tokens **identical** on every prompt. State diagnostics -
   `gdn_state`, `conv_ring`, the chunk's KV rows, the last hidden -
   recorded as max/mean relative difference. **Diagnostic, not gate**
   (spec-1 ruling: tokens gate, tensors diagnose), but the band is
   recorded and any growth across stages is a finding.
4. **Determinism.** `prefill` twice from a reset state → bitwise-identical
   state buffers and tokens. No fp atomics anywhere in the path; any
   `sycl-tla` configuration that uses split-K atomics is disallowed.
5. **Decode is untouched.** The decode gate rows (32.22 / 29.33 t/s at
   `2a7df0b`) re-measured once at the spec's final sha: inside day drift
   (≤ 0.09%) or the difference is explained.
6. `-Wall -Wextra -Werror`; `-cl-denorms-are-zero` forbidden; correctly-
   rounded div/sqrt default; the launch/module-count invariants of the
   decode list unchanged (774 / 19).

## 7. The gate

Record-grade (idle box, medians of 3) `b70-decode --pp 4096 --tg 256` on
the RTN checkpoint, reporting **device-side `pp` (loader-excluded,
first-token-inclusive) and the unchanged `tg`**:

- **≥ the ruled bar (§2):** record, `docs/BENCHMARKS` pp rows, README
  phase-1 row updated to "pp target met - device-side", tag `spec2-done`.
- **Short:** memo per the 1.5/1.7 pattern - per-stage yields, remaining gap,
  which term missed its Stage-0 price, redesign menu.

Either way the memo carries the **"skill or silicon" answer for prefill**:
where the composed ceiling sits against vLLM's 1973, term by term - and
which terms are Intel's code running at Intel's rate versus ours.

## 8. Constraints

- Box workflow `tools/box.sh`; JOBS 44; the **system** oneAPI toolchain
  (`/opt/intel/oneapi/compiler/2026.1/bin/icpx` via full path or
  `setvars.sh`, `docs/10-the-box.md:42`) - the container's SYCL is the
  operator's and stays read-only; `sycl-tla` built once as a static
  dependency, version pinned in the build.
- Probes run on card 1 (`ZE_AFFINITY_MASK=1`) while a server holds card 0;
  record rows only on a provably idle box (zero DRM holders).
- Every number labelled measured vs derived vs estimated/external, with
  grade and conditions; no two values for one quantity without a
  reconciliation sentence.
- Nothing pushed; tags local; work on the branch the operator names.
- Design must not preclude prefix caching: chunk boundaries at multiples
  of 1024 align with the queued block-snapshot scheme
  (`docs/04-architecture.md`, "Follow-on"); no work on it here.

## 9. Records

`docs/12-kernels.md` gains a prefill section (dequant scratch, GEMM
interface, attention, GDN chunk mechanism); `docs/15-step-anatomy.md` gains
the prefill-step anatomy; `docs/01-hardware.md` gains the **measured** XMX
rate; `docs/04-architecture.md` updated: the SYCL prefill path is now
real; `docs/06-prior-art.md` gains the vLLM→oneDNN→`gemmstone` call chain
**additively** (its `gemmstone` entry is already correct - §1);
`docs/BENCHMARKS.md` pp rows with both labels; `docs/05` prefill verdict.
`docs/07-open-questions.md` gains one: the box's driver/IGC stack (26.27 /
IGC 2.38) is **newer than `sycl-tla`'s validated Xe2 CI matrix** (26.01 /
IGC 2.27) - found by plan 6a, unpriced.

## 10. Out of scope

Tokenizer, chat template, HTTP/SSE (spec 3); MTP / speculative decode;
prefix caching / session continuation; continuous batching; fp8 KV;
**our own DPAS GEMM in OpenCL C** (the toolchain can, §1; it is the
recorded fallback, not a work item); **native int4 DPAS via inline vISA**
(a docs/01 note); MoE / other models.

## 11. Risks, stated

- **`sycl-tla` toolchain coupling** (icpx + SYCL runtime version vs the L0
  driver; build time). Mitigation: P1 first; pinned version; static link;
  our own kernels remain the fallback for every term.
- **FMHA does not instantiate at head_dim 256.** Mitigation: our flash
  kernel is designed and priced in P4's fallback branch; Task 5's GQA
  packing transfers.
- **Chunked-GDN rounding breaks token exactness on long prompts.**
  Mitigation: fp32 throughout; intra-chunk 64 matching the `fla`
  reference; P5 measures the state band before any gate is run; the
  self-consistency control localises divergence to a layer.
- **Memory at C = 4096** (scratch + activations + 16k KV). Mitigation:
  fallback C = 2048, priced.
- **The dequant-scratch overhead is larger than derived.** P3 measures it;
  if > 20% the design revisits a second weight layout for prefill (priced,
  memory-bounded).
- **Intel's CuTe GDN kernel does not fit our state layout.** Then S2
  writes ours (§3.4's bullet list is the design); P5 answers this in days,
  before S2 commits either way.
- **The composed ceiling lands under vLLM.** Then the memo says so with
  the per-term evidence - the same answer spec 1.7 gave for MBU.
