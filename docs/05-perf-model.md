# Performance model

## The roofline

Decode at batch 1 reads every weight once per token, so:

```
tokens/sec_ceiling = read_bandwidth / bytes_read_per_token
                   = 600 GB/s / W
```

`600 GB/s` is **measured** (doc 01). `W` for the phase-1 model is **measured
from the safetensors headers**, deduplicated through the index manifest
(doc 03): **15.52 GB per token** - 12.16 GB int4 weights, 0.76 GB f16 scales,
2.54 GB bf16 `lm_head`, 0.05 GB of norms and small bf16 projections. Embedding
rows are gathered (~0), `qzeros`/`g_idx` are dropped at load, the vision tower
is skipped, the MTP head is phase 2.

```
ceiling(27B, no MTP) = 600 / 15.52 = 38.7 t/s
vLLM p314-t214-vxkp0  =               31.50 t/s  ->  489 GB/s  ->  81% MBU
```

Resident bytes on load should equal this figure unless the loader pads; the
first load asserts it.

**Two constants, both measured, and which one to use.** The loader now reports
what it actually makes resident: **15,539,980,288 B = 15.540 GB per token**
(docs/13) - the 15.519 GB of doc-03 header arithmetic plus 0.016 GB of tiling
pad and 0.005 GB of fp32-widened norms. And doc 01 measures **590 GB/s** through
a plain Level Zero launch (`probe_bw`) against the 600 GB/s it recommends as the
general roofline denominator. Every number below and in
[BENCHMARKS.md](BENCHMARKS.md) uses **15.540 GB and 590 GB/s**, because that
pair describes the path this engine submits on, and it is applied to vLLM's
number too so the comparison is like-for-like:

```
ceiling(27B, no MTP) = 590 / 15.540 = 37.97 t/s   (26.34 ms/token)
vLLM p314-t214-vxkp0 =                31.50 t/s  ->  489 GB/s  ->  83.0% MBU
b70-decode 62bdd4d   =                23.73 t/s  ->  369 GB/s  ->  62.5% MBU
```

## Phase 1, measured - the honest verdict

**b70-decode does not beat vLLM. Measured 2026-08-25, median of three runs on
an idle box: 23.73 t/s at tg256, depth 4096 - against vLLM's 31.50. That is
24.7% short, a factor of 1.327 the wrong way.** Per token: 42.14 ms against
vLLM's 31.75 and a 26.34 ms roofline. Spread over the three runs was 0.02 t/s
(0.08%). Full rows, the depth experiment and the exact command are in
[BENCHMARKS.md](BENCHMARKS.md#b70-decode--this-project-phase-1).

Nothing was tuned to produce that number and nothing will be tuned in the plan
that measured it. What the plan owes instead is the decomposition, because the
decomposition is what scopes the next spec.

### Where the 42.14 ms goes

| part | ms/token | share | how it was obtained |
|---|---|---|---|
| GEMV - 257 of the 645 launches, the int4 mixers/MLPs and bf16 `lm_head` | **28.35** | 67.3% | **measured per kernel**, `probe_gemv` 2026-08-24, at exactly the shapes, layout and `S` the model table binds; summed over the layer counts |
| everything else - 388 launches (241 `prep`, 48 `gdn_step`, 48 `attn`, 48 `in_proj_a‖b` bf16 GEMV, `embed_gather`, 2× `argmax`) | **13.79** | 32.7% | **aggregate**: the measured step minus the row above |
| - of which `attn_decode`'s real per-block work at depth 4096 | 2.34 | 5.6% | **measured**, the depth experiment below |
| - of which `embed_gather` + `argmax` | 0.008 | 0.02% | **measured per kernel** (doc 12) |
| - of which the fixed attention grid's early-out | 0.046 | 0.11% | **measured**, the `--max-len` experiment below |
| - leaving `prep` + `gdn_step` + `attn_prep`/`attn_reduce` + the `a‖b` GEMV | **≈ 11.4** | 27.1% | **aggregate**, not separated: ~31 µs per launch over 369 launches |
| host, outside the fence entirely | **0.097** | 0.23% | **measured**, `Engine::last_gen_ms() − last_fence_ms()` |
| dispatch inside the fence (645 × 0.52 µs) | 0.335 | 0.8% | **estimated** from doc 07 #5 |

Read that table twice before proposing anything.

1. **The host-overhead thesis is finished for this engine.** 99.77% of the step
   is inside `execute` + `fence.wait()`; the host spends **97 µs per token**
   writing four bytes and submitting. There is no host work left to delete. The
   thesis was always about vLLM's Python/Triton launch path (doc 09), and
   replay did exactly what doc 08 said it would - which is precisely why the
   remaining gap is *not* excusable as overhead. It is kernel time.
2. **GEMV is not the problem.** 28.35 ms of measured GEMV is 89% of vLLM's
   *entire* 31.75 ms step. Those kernels run at 89-97% of 600 GB/s and the
   engine binds the probe's own best layout and `S` for every shape, so the
   "GEMV fill at the chosen `S` vs the probe matrix" suspect is **ruled out**:
   production and probe are the same configuration. There is ~2.0 ms of
   headroom between the GEMV sum and the 26.34 ms roofline, and that is all
   there is.
3. **The 13.79 ms of non-GEMV work is the whole gap.** To reach 31.75 ms/token
   the budget for everything that is not a GEMV is **3.4 ms**; we spend 13.79.
   Closing 10.4 of those milliseconds is exactly the phase-1 shortfall.
4. **The attention early-out is exonerated** (doc 07 #12, resolved): a grid
   sized for `max_len` 16384 costs **0.046 ms/token - 0.11% of the step** in
   idle work-groups. Context-bucketed lists would buy nothing.
5. **The named suspect inside the 11.4 ms is `prep_res_norm`**, on the strength
   of its own design note rather than a measurement: 129 launches per token, and
   each is a **single work-group** reducing 320 KB of split-K partials on one
   Xe-core (doc 12, "One work-group per token in `prep_res_norm` is the risk in
   this design, and it is unmeasured"). At 40 µs each that alone is 5.2 ms. The
   `in_proj_a‖b` GEMV is the second: N = 128 is 8 subgroups on a device with 32
   subslices (doc 01), 48 times per token, and it was never in the probe matrix.
   `gdn_step`'s 48 launches move 396 MB - 0.67 ms at the roofline - so if it is
   costing multiples of that, the cause is occupancy, not traffic.

### What spec 1.5 is scoped to do

In this order, because that is the order the evidence supports:

1. **Get a per-kernel profile.** The 11.4 ms is an aggregate and no amount of
   arguing splits it. Level Zero kernel timestamps on a one-off instrumented
   capture, or a `probe_prep` / `probe_gdn` in the shape of `probe_gemv`.
   Nothing else should be attempted before this number exists.
2. **`prep_res_norm`'s single work-group**, if the profile confirms it: a
   two-stage reduction, or fold the split-K sum into the GEMV epilogue.
   129 launches is where the leverage is.
3. **`in_proj_a‖b`**: 48 launches of an 8-subgroup kernel. Fusing it into the
   `qkv‖z` GEMV (doc 04's fusion item 4) removes the launch and the fill
   problem at once - the one fusion the measurement now argues for.
4. **`gdn_step` occupancy**, if the profile puts it above ~1 ms.
5. Only then the ~2.0 ms of GEMV headroom, and the `lm_head` quantisation below,
   which is worth 4.35 → ~1.1 ms and needs no kernel work at all.

Fusion for its own sake stays rejected: 645 × 0.52 µs = 0.335 ms is 0.8% of the
step (doc 07 #5, estimated). Kernel *count* is not the problem; what those
kernels do while they run is.

## Where the headroom actually is

The original claim was that the gap between the current stack and the ceiling is
**host overhead**, not kernel quality:

| Evidence | Reading |
|----------|---------|
| MXFP4 72.65 vs GPTQ-int4 73.31 t/s - two unrelated kernel paths within 1% | Not kernel-bound |
| Same model, XPU graphs off 10.55 vs on 31.5 t/s | ~3× of the cost is host-side, and graphs already recover most of it |
| Measured MBU 50-63% | The card is idle 40% of the time |

That evidence was gathered on the MoE models. Per model, with the 27B's `W`
now **measured** and the MoE `W` still an estimate (doc 07 #1):

| Model | t/s | `W` | Ceiling @ 600 GB/s | MBU |
|---|---|---|---|---|
| **Qwen3.8-27B dense int4 (phase 1)** | 31.50 | **15.52 GB, measured** | **38.7** | **81%** |
| Ornith 35B-A3B MoE, MXFP4 / GPTQ | 72.65 / 73.31 | ~3B active × 4.1 bit + lm_head ≈ 4-5 GB, estimate | ~120-150 | ~50-60% |

The "50-63% MBU" figure describes the MoE models. On the dense 27B vLLM is
already at 81%, so **the entire remaining gap to the roofline - host overhead,
device under-fill, everything - is ×1.23.** The MoE number also has an obvious
mechanism (160 launches and 200 allocations per step, doc 03) that the dense
path under XPU graphs does not; vLLM's residual per-step scheduling is a few ms
against a ~32 ms step, which is exactly consistent with 81%.

So the honest projection for phase 1 is a product of small factors, not one
large one:

| Lever | Ceiling effect | Est. t/s |
|---|---|---|
| vLLM today | - | 31.50 |
| Fill the device (GEMV + split-K) and delete host work: 81% → ~95% MBU | ×1.17 | ~36.8 |
| Drop `qzeros` + `g_idx` | ×1.013 | ~37.3 |
| `lm_head` int8 | ×1.09 | ~40.6 |
| `lm_head` int4 (instead of int8) | ×1.14 | **~42.5** |

**~1.35× over vLLM at best without speculation (≈42 t/s against a 44 t/s
int4-`lm_head` roofline).** Phase 2 is where larger numbers live: with MTP the
verify step runs at `M = k + 1` for one weight read, and the draft step reads
0.85 GB (MTP head) + 2.54 GB (`lm_head`) - quantising both to int4 cuts that to
~0.9 GB. At an assumed acceptance of ~0.7 per draft token (**not measured**; vLLM's
45.23 at 2 drafts implies at least that) the phase-2 ceiling is ~55 t/s with
bf16 `lm_head` and ~70 t/s with everything int4, against vLLM's 45.23.

- **MoE (phases 3-4):** the host thesis is the whole game - ~×1.7-2 from replay
  alone if MBU moves from ~55% to ~90%.

**Not 5×** in any case. Anyone promising more is counting the graphs-off number
as the baseline, which is not the fair comparison.

Set expectations there and the project stays honest. The other three goals -
learning, dependency control, shipping something - are what justify the effort
beyond the multiplier.

## Specialisations a general engine will not take

Ranked by expected value.

### 1. Quantise `lm_head`

`lm_head.weight` in phase 1 is **BF16, 248320 × 5120 = 2.54 GB** (measured),
read in full on every token - **16.4% of `W`**. At int4 it is 0.66 GB; at int8
1.27 GB.

It costs one offline conversion pass (`tools/`) and **does not depend on the
host thesis being true**: ×1.14 on the ceiling at int4 (38.7 → 44.0 t/s),
×1.09 at int8 (42.1). Smaller than it would have been on a 9B, but it is the
only lever that does not need a kernel. Accuracy risk is real (`lm_head` is
sensitive) and must be checked against the unquantised baseline (doc 07 #6).
In phase 2 it pays twice per step - the MTP head reads `lm_head` for its draft
logits - and the 0.85 GB bf16 MTP head itself is the same kind of target.

### 2. Pre-swizzle weights at load

Rearranging weights once at load into the exact order the XMX mainloop wants is
free at run time. Every CUDA backend does it; vLLM's XPU MXFP4 path does not
(`compressed_tensors_moe_w4a4_mxfp4.py:190` is a bare `pass`). Size the win by
measuring the mainloop's cost with and without the ideal layout.

### 3. Delete per-step allocations

`vllm-xpu-kernels`' MoE path allocates **5 tensors per layer per step** -
200 allocations per decode step at 40 layers. Static allocation makes this zero.
Matters most from phase 3 onward.

### 4. Fuse the MoE launches (phase 3+)

Their MoE issues 4 kernel launches per layer per step - 160 per step at 40
layers. Fusing gate/up and the activation is a known ~2× reduction in launches.

### 5. Exploit 24 MB of L2

Unusually large for this class of GPU. A 4096×4096 int4 tile is 8 MB, so
multi-layer weight residency is at least arguable. Measure before designing
around it.

## What to measure, in order

1. ✅ **`W`** - 15.52 GB per token from the headers (doc 03). Confirm resident
   bytes on first load.
2. ✅ **vLLM baseline t/s** - pp4096 1973 / tg256 31.50, `p314-t214-vxkp0`, no
   speculation (BENCHMARKS.md). MTP: 42.56 / 45.23 at 1 / 2 drafts.
3. ✅ **Achieved MBU** - 81% on this model. The 50-63% figure is MoE-only.
4. ⚠️ **GDN vs GEMM time split** - **half-answered on our own engine
   2026-08-25** (doc 07 #3): GEMV **28.35 ms of a 42.14 ms step (67.3%)**,
   everything else 13.79 ms (32.7%). GEMM dominates as predicted. The estimate's
   *mechanism* was wrong: the non-GEMV third is not kernel count (645 × 0.52 µs
   = 0.335 ms, 2.4% of it) but time inside `prep` / `gdn_step` / `attn`, and
   this measurement does not separate those three. Doing so is spec 1.5's first
   task. **Estimate before measuring** (2026-08-22, kept for the record): the
   recurrent state is 3 MB per layer, read and written once per token - ~150 MB
   across 48 layers, ~2% of `W`. Expect GEMM to dominate bandwidth and GDN to
   dominate *kernel count*; under replay the second is what the fusion list in
   doc 04 attacks. If a profile of vLLM says GDN dominates, suspect its Triton
   launch path
   (`vllm/third_party/flash_linear_attention/ops/fused_recurrent.py`), not the
   arithmetic.
5. ✅ **Per-kernel fixed cost inside a replayed list** - **0.52 µs/kernel**
   (`noop`) and **0.63 µs/kernel** (`ctrl_read`) at N = 700, measured
   2026-08-22 (doc 07 #5, `tools/probe/probe_replay`). The empty submit +
   fence round trip - the floor no fusion removes - is **6.4 µs**. Both are
   under the 1 µs threshold, so the fusion list in doc 04 is *not* on the
   phase-1 critical path.
6. ✅ **GEMV bandwidth matrix** - the production `gemv.cl` at the five int4
   shapes the 27B runs after load-time fusion, both canonical layouts,
   `S` ∈ {1,2,4,8,16}, plus the bf16 `lm_head`: 51 configurations, measured
   **2026-08-24** with `tools/probe/probe_gemv` (re-run that day at the
   corrected `out/o_proj` shape 6144 × 5120 - spec §4.2 had said 5120 × 5120,
   which the model does not contain; doc 12). GB/s counts weight bytes
   (nibbles + f16 scales); each configuration is 40 launches in one replayed
   list cycling ≥ 72 MB of distinct weight copies, so the 24 MB L2 is missed
   and the number is DRAM bandwidth. **All 50 int4 configurations matched the
   CPU reference.** Best row per shape:

   | shape | K×N | layout | S | GB/s | % of 600 |
   |---|---|---|---|---|---|
   | out/o_proj | 6144×5120 | 0 | 4 | 549 | 92% |
   | q‖k‖v | 5120×14336 | 0 | 2 | 577 | 96% |
   | qkv‖z | 5120×16384 | **1** | 1 | 559 | 93% |
   | gate‖up | 5120×34816 | 0 | 8 | 555 | 93% |
   | down | 17408×5120 | 0 | 4 | 567 | 94% |
   | `lm_head` bf16 | 5120×248320 | tiled | - | **584** | **97%** |

   The probe's decision rule (spec §4.2) picks **layout 1** - by 1.7% summed
   over the five shapes, 3.6% in wall time - and `S` = 16 / 1 / 1 / 4 / 16 for
   the shapes in the order above. Doc 12 carries the full reasoning, including
   what layout 1 costs on the four shapes where layout 0 was ahead.

   Two conclusions. **The split-K argument in doc 08 is confirmed:** at
   N = 5120 - 320 subgroups, the worst fill case - the kernel reaches
   259 GB/s (43%) at `S` = 1 and 533 GB/s (89%) at `S` = 16, a 2.06× gain, and
   the same doubling appears on `down`. **`lm_head` needs no split-K:** 15520
   subgroups already reach 97% of the 600 GB/s denominator, which is 99% of the
   590 GB/s `probe_bw` measures through the same launch path. At that rate
   `lm_head` alone is 4.35 ms of a 25.8 ms token.

   ```bash
   tools/box.sh run "./build/tools/probe/probe_gemv" | tee docs/probe-gemv-2026-08-24.md
   ```

   The full 51-row matrix is committed verbatim as
   [probe-gemv-2026-08-24.md](probe-gemv-2026-08-24.md) (the superseded
   2026-08-23 run is kept alongside it). All of it is **M = 1**;
   the `S` picks are M = 1 picks (doc 12).

1-3, 5 and 6 are done; 4 is half-done as of 2026-08-25 (GEMM 67.3% / rest
32.7%, the "rest" not yet split three ways). Finishing it - a per-kernel
profile of `prep`, `gdn_step` and `attn` - is now the top item of spec 1.5,
because that aggregate 13.79 ms is the entire gap to vLLM.

## Benchmark

```bash
uvx llama-benchy --base-url http://0.0.0.0:8000/v1 --model "$MODEL" \
  --pp 4096 --tg 256 --concurrency 1 --depth 1 \
  --no-cache --exact-tg --latency-mode generation
```

Reference numbers from the vLLM stack (image `p314-t214-vxkp0`, torch 2.14,
graphs on unless noted):

| Model | Config | pp4096 | tg256 |
|-------|--------|--------|-------|
| Qwen3.8-27B dense int4 | - | 1973 | 31.50 |
| Qwen3.8-27B dense int4 | MTP, 1 draft | 1931 | 42.56 |
| Qwen3.8-27B dense int4 | MTP, 2 draft | 1918 | 45.23 |
| Qwen3.8-27B MixedInt4 | **graphs off** | 2275 | **10.55** |
| Ornith MXFP4 MoE | - | ~9250 | 72.65 |
| Ornith GPTQ-int4 MoE | - | ~9232 | 73.31 |

Phase-1 baseline is the first row: **pp4096 1973 / tg256 31.50.** Full tables,
images and the exact serve and bench commands are in
[BENCHMARKS.md](BENCHMARKS.md); the dense 27B rows there use `--max-model-len
16k`, `pp4096`, `depth 1`, `concurrency 1`, which is the v1 definition of done.

**b70-decode's own row, measured 2026-08-25: tg256 23.73 t/s at depth 4096** -
`tools/bench_decode.sh`, median of three, and 24.7% short of that baseline. No
`pp4096` figure exists yet: this engine has no prefill kernel (spec 2), so a
prompt costs one decode replay per id - 4096 ids in 170.5 s.
