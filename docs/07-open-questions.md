# Open questions

Ranked by how much the answer would change the design. Nothing here is settled -
do not let an estimate from these docs become an assumption in a spec.

## 1. What is `W`, the bytes read per decode token? - **resolved for phase 1**

**15.52 GB per token** for `Vishva007/Qwen3.8-27B`, measured 2026-08-22 from
the safetensors headers with the index as the manifest (doc 03): 12.163 GB
int4 `qweight`, 0.760 GB f16 scales, 2.543 GB bf16 `lm_head`, 0.052 GB bf16
small tensors. `qzeros` (0.190) and `g_idx` (0.012) are dropped at load; the
embedding is gathered; the vision tower is skipped. The checkpoint has no
duplicated tensor names, so header arithmetic is exact here - the
dedup-by-name rule stays in the loader for checkpoints that do.

Reproduce: `python3 tools/probe/checkpoint_bytes.py <snapshot>`.

Consequence: vLLM's 31.50 t/s is **81% MBU**. The host-overhead thesis is
nearly exhausted on this model (doc 05); it remains the main lever for the MoE
phases, where `W` is still an estimate. Confirm resident bytes on first load.

## 2. The phase-1 model does not currently load in vLLM - **resolved by re-scoping (#14)**

Phase 1 is now the 27B, which loads and has a baseline. The loader
requirements at the end of this entry still stand unchanged - the 27B ships
the identical `model.language_model.*` / `model.visual.*` / top-level
`lm_head.weight` layout. Kept for the record:

`letechlead/Ornith-1.5-9B-INT4-W4A16-AutoRound` declares
`architectures: ["Qwen3_5ForConditionalGeneration"]` and ships tensors named
`model.language_model.*` (2177), `model.visual.*` (549), `lm_head.weight` (1).

vLLM (`v0.27.2rc1.dev342+g0a5a55136`, image `p314-t214-vxkp0`) instantiates
`Qwen3_5Model`, whose parameters are `layers.*`, and fails:

```
ValueError: There is no module or parameter named 'language_model' in Qwen3_5Model
```

Fails identically with and without `--language-model-only`.

Two consequences:

- **The "beat vLLM" baseline for phase 1 has no number yet.** Either fix the vLLM
  side, or pick a phase-1 checkpoint that loads, or accept the 27B numbers as the
  reference and re-scope phase 1.
- **Our loader must handle this layout natively**: strip the
  `model.language_model.` prefix, skip `model.visual.*` entirely, and treat
  `lm_head.weight` as top-level. Design for it rather than requiring
  re-exported checkpoints.

## 3. Is GDN or GEMM the larger share of decode time? - **resolved: GEMM, 68.8%; GDN's own kernel is 1.7%**

48 of 64 layers are linear attention, and GDN is the one kernel with no
`sycl-tla` starting point. If GDN dominates, it should be the first kernel
written and the first optimised. If GEMM dominates despite GDN being 48/64
layers, the decode GEMV carries more of the outcome than the layer count
suggests.

Measure on the current stack with a profiler before choosing what to build first.

**Estimate (2026-08-22):** GDN state is 2 MB per layer (doc 03), ~100 MB per
token across 24 layers - ~2% of `W`. GDN should be a kernel-count problem, not
a bandwidth one, and replay removes most of the kernel-count cost. Expect GEMM
to dominate. If a vLLM profile says otherwise, the Triton launch path is the
first suspect.

**Half-answered on our own engine, 2026-08-25 - and the estimate above is
wrong about *why*.** Of a measured 42.14 ms step (doc 05), GEMV is **28.35 ms
(67.3%)** - measured per kernel by `probe_gemv` at the shapes the model table
binds - and everything else is **13.795 ms (32.7%)**. So GEMM does dominate, as
predicted. What was not predicted is that the non-GEMV third is **not** a
kernel-count problem: 645 launches × 0.52 µs is 0.335 ms, 2.4% of that 13.795 ms.
It is time spent *inside* `prep`, `gdn_step` and `attn`, and this measurement
does **not** separate the three - that separation is spec 1.5's first job
(doc 05, "What spec 1.5 is scoped to do"). Attention's share is bounded:
2.31 ms of per-block work at depth 4096 - *estimated* by extrapolating the #12
experiment's two points, not timed.

**Resolved, 2026-08-25, per kernel in situ** (`b70-decode --profile`, the full
anatomy in [15-step-anatomy.md](15-step-anatomy.md)). GEMM dominates and the
margin is slightly larger than the transplant said: **29.005 ms, 68.8%**, with
everything else at 13.136 ms. The three-way separation the paragraph above
deferred:

| | ms/token | share | |
|---|---|---|---|
| GEMV (257 launches) | 29.005 | 68.8% | measured, in situ |
| `attn_decode` (16) | 5.782 | 13.7% | measured, in situ - 2.45× the 2.31 ms estimated above |
| `prep` (241) | 3.573 | 8.5% | measured, in situ |
| `a‖b` GEMV (48) | 2.335 | 5.5% | measured, in situ |
| **`gdn_step` (48)** | **0.733** | **1.7%** | measured, in situ - 92% of device bandwidth |

The 2026-08-22 estimate was right that GDN is a small share and wrong about the
reason it would be small: not "kernel-count, and replay removes it" but *it is
the best-occupied kernel in the engine* - 192 work-groups at 540 GB/s, 1.09× its
own traffic floor (doc 12, `gdn_step` → Measured). The occupancy problem the
estimate expected to find in GDN is real and lives in the kernels that were
given one or two work-groups - but **the unit is the subgroup, not the
work-group**: spec 1.5's lever L2 gave `a‖b` 4× the work-groups at an unchanged
subgroup count and it bought nothing, and a 16-way K split (8 → 128 subgroups)
took that run's row from 2.341 to **0.256 ms/token** (docs/15 §L2). The `a‖b`
row above is the pre-lever number as *this* table's run measured it, 2.335 -
0.3% from the L2 before-run's 2.341, which is the instrument's spread on this
row and the reason the two are never mixed inside one sentence.

## 4. What is vLLM's MBU on the phase-1 model? - **resolved: 81%**

31.50 t/s × 15.52 GB = 489 GB/s of 600. The 50-63% figure was MoE-only. The
projected headroom for phase 1 is therefore ×1.23 to the bf16-`lm_head`
roofline and ~×1.35 with `lm_head` at int4 (doc 05) - not 1.4-1.5× from host
work alone. The thesis survives; its weight shifts to the MoE phases and to
phase 2.

**On the loader-measured `W` and the Level Zero bandwidth (15.540 GB, 590 GB/s
- the pair docs/05 and BENCHMARKS.md use from 2026-08-25 on): 489 GB/s of 590 =
83.0%.** Same conclusion, and the denominator that b70-decode's own 62.5% is
quoted against, so the two are comparable.

## 5. What is the per-kernel fixed cost inside a replayed command list?

Measured 2026-08-22 by `tools/probe/probe_replay`: one in-order regular list
of N launches, closed once, replayed 1000 times behind a fence (median, after
20 warm-ups).

| kernel | N | us/replay | us/kernel |
|---|---|---|---|
| (empty list) | 0 | 6.4 | - |
| noop | 1 | 9.9 | 9.89 |
| noop | 250 | 134.6 | 0.54 |
| noop | 700 | 360.7 | 0.52 |
| ctrl_read | 1 | 5.9 | 5.90 |
| ctrl_read | 250 | 161.4 | 0.65 |
| ctrl_read | 700 | 438.6 | 0.63 |

Per-kernel cost is 0.52 µs (`noop`) and 0.63 µs (`ctrl_read`, which reads one
dword of a shared-memory control block - the shape of every decode kernel's
first instruction) → **< 1 µs, so fusion is not on the phase-1 critical path**
(spec 1 §4.1 rule: ≥ 3 µs yes, < 1 µs no) and the unfused ~645-kernel list
(spec 1 §9.1) ships first at 0.335 ms of a step **measured at 42.14 ms**
2026-08-25 - **0.8%**, confirming the call (doc 05).

The empty submit + fence round trip is **6.4 µs** - the floor no fusion
removes, paid once per token. The N = 1 rows are that floor plus one kernel,
not a per-kernel number.

**Measured in situ, 2026-08-25, on the real 645-kernel decode list** - the
numbers above are noop and `ctrl_read` lists; these are the engine's own step
(`b70-decode --profile --depth 4096 --steps 32`, [15](15-step-anatomy.md)):

| | µs/step | µs/launch | kind |
|---|---|---|---|
| profiled gap (fence wall − Σ kernel durations) | 851.4 | 1.320 | measured - **upper bound** |
| **un-instrumented gap** | **473** | **0.733** | **derived** |
| this table's `noop` floor × 645 | 335 | 0.52 | estimated |

The profiled number is an upper bound because every launch in a profiled list
signals a host-visible event, and that flush is inside the *gap* (it lands after
`kernelEnd`, so it never touches a kernel's own duration). The **derived** row
is the honest one and it is arithmetic over two instruments: the bench step is
42.141 ms of which 0.097 ms is host outside the fence, so an un-instrumented
fence is 42.044 ms; Σ of the in-situ kernel durations is 41.571 ms; the
difference is 0.473 ms.

**0.733 µs/launch lands between this table's `noop` (0.52) and `ctrl_read`
(0.63) floors, a shade above both** - which is what 645 kernels that each read
the shared control block should cost, and it confirms the probe's
transplantability at the third decimal. The conclusion is unchanged and now
rests on a measurement rather than an extrapolation: **0.473 ms is 1.1% of a
42.141 ms step, so fusion for launch-count's sake is not a lever** (spec 1 §4.1's
≥ 3 µs rule). A by-product: the profiler's own distortion is 0.379 ms/step,
**0.587 µs per launch** (derived) - a profiled step is 0.9% longer than a real
one, all of it in the gap.

## 6. How much accuracy does quantising `lm_head` cost?

On the 27B it is 2.54 GB → 0.66 GB per token at int4 (×1.14 on the ceiling)
or 1.27 GB at int8 (×1.09) - doc 05. `lm_head` is known to be
quantisation-sensitive and the checkpoint author chose not to (`lm_head:
false` in `quantization_config`). Needs a perplexity or `lm_eval` comparison
against the unquantised baseline before adoption. In phase 2 the same question
applies to the 0.85 GB bf16 MTP head, and `lm_head` is read twice per step.

## 7. Can 24 MB of L2 be exploited deliberately?

Unusually large. A 4096×4096 int4 tile is 8 MB. Whether multi-layer weight
residency is achievable, or whether the replacement policy defeats it, is
unknown. Measure before designing around it.

## 8. Is `MainloopIntelW8A8` worth it for prefill?

int8 XMX has roughly 2× bf16 throughput and prefill *is* compute-bound. But it
requires int8 activations, i.e. dynamic activation quantisation and its accuracy
cost. Out of scope for phase 1; revisit once decode is winning.

**Largely superseded by #9.** W4A8 gets the same int8 systolic path while keeping
weights at 4 bits - same compute win, half the memory traffic, and no need to
re-quantise the checkpoint. Answer #9 first; if it holds, W8A8 is uninteresting.

## 9. Is native mixed s8 × s4 DPAS reachable, and what does it cost?

`oneDNN/src/gpu/intel/gemm/jit/pd.cpp:772` documents a **native int8 × int4 DPAS**
on all pre-Xe3p hardware - Battlemage included - gated only on the int4 matrix
having no zero points. Our checkpoints are all symmetric, so the gate opens.

If real, a **W4A8 prefill** path gets int8 systolic throughput *and* zero
dequantisation, which neither vLLM nor OpenVINO uses (both run W4A16). That would
beat both rather than matching one.

**Partly answered (2026-08-22):** `sycl-tla` *does* declare the atoms -
`include/cute/arch/mma_xe.hpp:287-297`, `XE_DPAS_TT(d, s8, s4, d)` plus every
u/s permutation, compiled out only for CRI. What it lacks is a mainloop that
feeds them: `xe_mma_mixed_input.hpp` widens int4 to 16-bit. So the question is
no longer "is it reachable" but "write a CuTe mainloop on an existing atom, and
does it win".

Still unknown:

- Whether the s8×s4 DPAS runs at the int8 rate on a B70 (it might be
  microcoded to the same cost as s8×s8 plus a widen - measure with a single
  atom in a loop before writing a mainloop).
- What dynamic int8 activation quantisation costs in output quality.
- Whether the win survives the quantise-activations step, which is itself a pass
  over the activations.

Cheapest probe: benchmark oneDNN directly at prefill shapes, W4A16 vs W4A8, before
building anything.

## 10. Does MXFP4 need original kernel work on Battlemage?

`sycl-tla`'s block-scaled examples are `50_xe35_*` / `51_xe35_*` - Xe3.5, not
BMG. Phase 3 may need to write or adapt block-scale handling rather than lift it.
Confirm by attempting to instantiate the block-scaled mainloop for a BMG target
before committing to the phase-3 schedule.

## 11. Which tokenizer dependency, and what does it cost goal 3?

Not mentioned in the first draft of these docs at all. Every request passes
through BPE encode; every token through detokenise. Options and trade-offs in
[11-tokenizer-and-chat-template.md](11-tokenizer-and-chat-template.md). The
decision is between parity (HF `tokenizers` via FFI - exact, but a Rust
toolchain in the build) and ownership (a hand-written byte-level BPE - no
dependency, parity must be proven on a corpus). Resolve before the server
milestone, not before the decode core.

## 12. What does a fixed-grid attention kernel cost at short context? - **resolved: 0.046 ms/token, 0.11% of a step. Keep the fixed grid.**

Under replay the attention grid is sized for `max_model_len` and idle
work-groups exit early (doc 04, "Attention under replay"). The original test:
measure the step at `seq_len = 64` vs `seq_len = 4096` with the same list, and
if the difference exceeds ~2% of a step, capture context-bucketed lists
instead.

**Measured 2026-08-25**, `b70-decode --bench`, tg 256, three runs each on an
idle box (BENCHMARKS.md carries every row):

Live blocks are `nb = (pos + m)/ATTN_BLOCK + 1` (`attn.cl:551`; the divisor was
a literal 256 when this was measured - see the L5 note at the end of this
entry), counted over the 256 positions each `tg` walks:

| shape | grid (`attn_decode`) | live blocks (`nb`) | t/s | ms/token |
|---|---|---|---|---|
| depth 4096, `--max-len 16384` | 4 × 64 | 17 throughout (`pos` 4096…4351) | 23.73 | 42.141 |
| depth 64, `--max-len 16384` | 4 × 64 | mean 1.25 (1 for 192 tokens, 2 for 64) | 25.00 | 39.997 |
| depth 64, `--max-len 4096` | 4 × 16 | mean 1.25, same as above | 25.03 | 39.951 |

The first two rows are the experiment as written: **2.144 ms/token, 5.1% of the
step**, well over the 2% bar. But that difference is not what the question was
asking about - it is the *real* work of 15.75 more live 256-position blocks,
which a bucketed list would still have to do. Reading it as the fixed grid's
cost would have got the answer exactly backwards.

The third row is the experiment the question actually needed, and it exists
because `attn_decode`/`attn_reduce` are already compiled at `MAXLEN = 4096` as
well as 16384: **same depth, same live blocks, same work, one quarter of the
grid.** The difference is **0.046 ms/token - 0.11% of the step** for 48 extra
idle blocks × 4 kv-heads × 16 layers = **3072 extra work-groups**, i.e. **~15 ns
per early-outed work-group**. (`--max-len` also changes `attn_part`'s stride and
the KV footprint, but `attn_reduce`'s merge loop is bounded by `nb` - 1.25 in
both rows - so the early-out dominates the difference; doc 12's `attn` →
Measured spells this out.) That is the number this entry wanted.

The pair is also **thermally matched** - both runs follow the same 2.4 s ingest
- which the 2.144 ms depth delta is not: its depth-4096 run follows 170 s of
continuous replay. That asymmetry is one reason doc 05 labels the 2.31 ms of
attention work it extrapolates from that delta *estimated* rather than
measured.

**Re-checked against a 4× larger grid, 2026-08-25 (spec 1.5 lever L5) - the
resolution survives, and the margin is wider than it was.** Every number above
was measured at `ATTN_BLOCK` 256, where `attn_decode`'s grid at `max_len` 16384
was 4 × 64 = 256 work-groups. L5 took the block to **64**, so the grid is now
4 × 256 = **1024** and the idle work-groups at depth 4096 went 188 → 764 per
launch. That is exactly the input this entry's conclusion depends on, so it is
worth saying explicitly what happened to it: **nothing bad.** The per-launch
in-situ timing of the retiled kernel is 224.046 µs against 369.988 (docs/15
§L5), i.e. the launch got **39% faster while quadrupling its grid** - which is a
far stronger version of this entry's finding than the 0.11% it was resolved on.
**The transplant, done once and properly** (an earlier version of this
paragraph mixed two different rates and got it wrong by 47×). This entry's own
rate is **0.046 ms/token / 3072 work-groups = 14.97 ns per early-outed
work-group**. The retile adds 576 idle work-groups per launch, and there are 16
`attn_decode` launches per token, so it adds **9216 per token**:

    9216 x 14.97 ns = 138 us/token = 0.138 ms  ->  0.38% of the 36.32 ms step
    per launch: 576 x 14.97 ns = 8.6 us        ->  3.8% of the retiled 224.046 us

That is the **conservative** rate. docs/15 §2 measured the same experiment on the
kernel itself rather than on a whole-step bench difference and got 0.04% of a
153.608 µs launch for 192 work-groups = **0.32 ns/work-group**, 47× smaller,
which would put the same 9216 at 0.003 ms/token. The two disagree because one is
a bench delta carrying everything else that differs between two runs and the
other is a kernel timing; this entry keeps its own, larger number so the bound is
not flattered.

**Neither figure is a term to add, and that is the point.** The retiled kernel's
224.046 µs/launch was measured *with* all 764 of its idle work-groups in the
grid, so whatever they cost is already inside the −2.219 ms/token the lever
banked. The transplant only answers "did quadrupling the grid blow the early-out
budget", and the direct measurement answers it better: **the launch got 39%
faster while its grid grew 4×.** **The fixed grid is cheaper than this entry
could prove in 2026-08's measurement, not more expensive.**

**Resolution: the fixed grid stays.** Context-bucketed lists would buy 0.11% and
cost a captured list per bucket, the memory for it, and a host-side branch on
context length in the one loop that currently has no branches at all. The
early-out (doc 12, "The early-out - the answer to a grid that cannot be
re-sized") does its job. Attention is not on the phase-1 critical path - doc 05
puts the phase-1 shortfall in `prep`/`gdn_step` instead.

```bash
tools/bench_decode.sh                     # row 1
tools/bench_decode.sh --depth 64          # row 2
# Row 3 has no --max-len flag in the harness, so it is a direct box.sh run. The
# $(...) is deliberately substituted by the LOCAL shell: tools/box.sh syncs the
# tree without .git, so the box cannot name the commit and the row would print
# `unknown` (the run recorded above passed B70_GIT_SHA=62bdd4d this way).
SHA=$(git rev-parse --short HEAD)
tools/box.sh run "B70_GIT_SHA=$SHA ./build/src/cli/b70-decode <model> --bench --depth 64 --tg 256 --max-len 4096"
```

## 13. Can a `sycl-tla` kernel be appended to a raw L0 command list at all?

Doc 04 recommends keeping `sycl-tla` out of the decode list, which sidesteps
this. But phase 3's grouped GEMM may want a CuTe kernel at decode. Probe:
take `00_bmg_gemm`, get its `ze_kernel_handle_t` via
`sycl::get_native<backend::ext_oneapi_level_zero>` on a named kernel bundle,
append it to a regular command list, replay twice, diff. Cheap, and it settles
whether the two toolchains can ever share a list.

## 14. Which checkpoint is phase 1? - **resolved: `Vishva007/Qwen3.8-27B-W4A16-AutoRound-GPTQ`**

No working 9B exists on the box (2026-08-22). Phases 1 and 2 share the 27B;
its architecture, byte accounting and baseline are in doc 03 and
BENCHMARKS.md. Open sub-question: the implied MTP acceptance rate on this
model is unknown - vLLM's 42.56 / 45.23 at 1 / 2 drafts bounds it from below
but does not separate acceptance from step cost. Log it from vLLM
(`--speculative-config` metrics) before sizing the phase-2 ceiling.
