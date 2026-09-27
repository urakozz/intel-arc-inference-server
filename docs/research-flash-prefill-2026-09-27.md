# Research: a faster `pf_flash_attn` prefill kernel on the B70 (2026-09-27)

Question: how do we make the fused flash-attention prefill kernel
(`src/kernels/prefill/pf_flash_attn.cl`, spec 6's `pfa_KT64_R16_H6_Q0`) substantially
faster? Today it is 116.3 ms for pp4096's attention phase (vLLM's flash kernel: 63.1 ms,
spec 6's F1 target: 80 ms) and 28 TFLOP/s = 15 % of the 183.45 TFLOP/s bf16 peak at depth.

Every number is **measured** or **derived** (arithmetic on measured numbers), or
**estimated** (a model, stated with its assumptions). "Static" means read from the Xe2
assembly that `ocloc` emits (IGC shader dump, `bmg-g31`, 256 GRF, the build's flags); it
costs no GPU time and says nothing about speed on its own.

**Headline.**

1. **Our kernel spills, and `exp` is the cause.** The
   main loop is 1639 instructions per 128 `dpas.8x8`; 9 of the 16 O accumulator vectors
   (4.9 KB per sub-group) go to scratch and back on every KV tile. The spill is caused by
   `exp()`: IGC lowers each `exp` to two `math.exp` plus a range-splitting sequence (88
   `mad`, 40 `rndz`, more `sel`/`cmp`), and those temporaries are what push the loop past
   256 GRF. **Switching to `exp2` with `ATTN_SCALE * log2(e)` folded in removes the spill
   entirely and cuts the loop to 1137 instructions** (static). `exp2` and `native_exp2`
   compile to the *same* code here (one `math.exp`, which is base-2 in the ISA), so there is
   no precision trade between them on this compiler.
2. **Measured: two one-line changes make the kernel 1.42x faster at every depth**, with
   the fp64 cosine unchanged: `exp2` with the folded scale (1.17-1.23x alone), and
   **8 query rows per work-group instead of 16** (6 sub-groups per WG instead of 12, so an
   Xe-core holds 30 of its 32 thread slots instead of 24; 1.14-1.27x alone). Together:
   **pfa_KT64_R8_H6 + exp2 = 39.5 ms against 56.1 at pos 30720 (40.4 TFLOP/s, 22 % of
   peak), 3.42 against 4.87 ms at pos 2048, 1.20 against 1.71 at pos 0.** Scaled onto the
   engine's 116.3 ms that is **about 82 ms for pp4096's attention** (derived), at F1's
   80 ms bar; vLLM is 63.1.
3. **The softmax-side levers are measured neutral once the spill is gone**: causal mask
   only on diagonal tiles, FA-4's conditional rescale, hardware bf16 conversion, deferred
   row sums and reversed row order each land within -3 % .. +1 %. The kernel is not
   issue-bound any more; what is left is memory-side.
4. **The structural ceiling is operand bandwidth, not math.** With 8 query rows per
   sub-group (ours and vLLM's), every K and V DPAS B-operand is used by exactly one
   `dpas.8x8`: 576 B of operands per DPAS against the production GEMM's 192 B. Past
   ~vLLM's level the kernel needs 16 (or 32) rows per sub-group, which for head_dim 256
   forces **splitting O over d across sub-groups and exchanging P through SLM** (what
   oneDNN's micro-SDPA and IPEX's XeTLA do for head 256; vLLM and sycl-tla do not).

## 0. Measured: the lever arms (2026-09-27)

`tools/probe/pfa_levers_research.cl` (the production source with each lever behind a bit
of `QREG`) run through the unchanged `probe_flash_attn` harness: random bf16 Q/K/V in the
production layouts, the fp64 CPU reference on 25 rows x 24 heads, bar cos >= 0.99999,
composed path as control, 20-iteration warm-up then 11 interleaved rounds, L0 kernel
timestamps. Device 0, `flock ~/b70-gpu.lock`. **Diagnostic grade, not record grade**:
device 1 was held by another agent's probe and the host load average was ~10 from other
builds; the paired ratios are the numbers to trust. Two sessions; the second (the table)
re-ran the control arms beside the R8 arms.

Arm names: `pfa_KT<kv tile>_R<rows per WG>_H6_Q<levers>`; levers 0 = production text,
2 = exp2, 47 = exp2+DM+LR+CVT+DS, 63 = 47+RV. Median ms (ratio against the production
arm `KT64_R16_Q0`, derived from the medians).

| arm | SGs / WG | pos 0 (C 2048) | pos 2048 | pos 30720 | TFLOP/s at 30720 | worst cos (all cases) |
|---|---:|---:|---:|---:|---:|---:|
| composed path (control) | - | 2.126 | 4.966 | 69.462 | 23.0 | 0.999996452 |
| **KT64_R16_Q0 = production** | 12 | 1.708 | 4.866 | 56.099 | 28.5 | 0.999998097 |
| KT64_R16_Q2 (exp2) | 12 | 1.383 (1.23x) | 3.966 (1.23x) | 47.928 (1.17x) | 33.3 | 0.999998097 |
| KT64_R8_Q0 (R8) | 6 | 1.484 (1.15x) | 4.253 (1.14x) | 44.267 (1.27x) | 36.1 | 0.999998097 |
| **KT64_R8_Q2 (exp2 + R8)** | 6 | **1.200 (1.42x)** | **3.420 (1.42x)** | **39.540 (1.42x)** | **40.4** | 0.999998097 |
| KT64_R8_Q47 (+DM LR CVT DS) | 6 | 1.221 | 3.515 | 40.858 | 39.1 | 0.999997414 |
| KT64_R8_Q63 (+RV) | 6 | 1.252 | 3.518 | 40.261 | 39.7 | 0.999997414 |
| KT32_R8_Q2 | 6 | 1.242 | 3.600 | 40.874 | 39.1 | 0.999998088 |
| KT32_R8_Q63 | 6 | 1.244 | 3.521 | 40.785 | 39.2 | 0.999997252 |

First session (same harness, the R16 lever ladder; ms at pos 0 / 2048 / 30720): Q0
1.626 / 4.674 / 56.155; Q2 1.353 / 3.806 / 48.002; Q11 (exp2+DM+CVT) 1.375 / 3.912 /
50.162; Q15 (+LR) 1.399 / 3.956 / 50.865; Q47 (+DS) 1.373 / 3.881 / 50.047; Q63 (+RV)
1.413 / 4.022 / 49.611; KT32_R16_Q63 1.429 / 4.058 / 45.664; KT64_R8_Q63 1.210 / 3.391 /
39.991. Q127 (the 32-row K load) **fails** at every case (cos 0.15-0.24): the
`transpose_32b_32r8x1c` register layout is not "rows 0-15 then 16-31"; unresolved.

Correctness: every other arm passes on (30720, 2048), (2048, 2048), (0, 2048),
(0, 2048, qscale 30) and (777, 300); pad rows finite everywhere. **exp2 leaves the worst
cosine bit-identical** to the production kernel's on every case (0.999998097,
0.999998232, 0.999998128, 0.999999009, 0.999998227); LR moves it in the seventh digit
(0.999997414 worst), still above the composed path's own worst.

Derived:

- **pp4096 attention**: 16 FA layers x (pos 0 + pos 2048 chunk) = 105.2 ms for the
  production arm and **73.9 ms** for KT64_R8_Q2 in this harness, ratio 0.703; applied to
  the engine's measured 116.3 ms, **~82 ms** (F1: 80, vLLM 63.1).
- **Depth**: 1.42x at pos 30720 too. With attention at 51.3 % of pp65536's GPU time,
  pp65536 would go from 1121.74 to ~1320 t/s (estimated, 0.487 + 0.513 / 1.42 = 0.848 of
  the time), and pp131072 by at least as much.
- **Numerics rule.** Spec 6 says "`exp`, never `native_exp`". The dump shows OpenCL's
  full-precision `exp2()` compiles to exactly what `native_exp2()` does (one `math.exp`),
  so the promotion should use `exp2(s * (ATTN_SCALE * M_LOG2E_F) - m)`: no `native_`
  builtin, and the K1 cosines above are unchanged. This still wants the operator's
  ruling, since the rounding point moves (scale folded before the subtraction).


## 1. Where the current kernel loses time

### 1.1 The work, per sub-group per 64-key tile (derived from the source)

| item | per SG-iteration | note |
|---|---:|---|
| `dpas.8x8` | 128 (64 QK + 64 PV) | 16 EU-cycles each: 2048 XMX cycles |
| K loads, `transpose_32b_16r8x1c` | 64 x 512 B = 32 KB | one message per QK DPAS |
| V loads, `transform_16b_32r16x2c` | 16 x 2 KB = 32 KB | |
| Q loads (Q0: re-read per tile) | 8 x 512 B = 4 KB | |
| spill traffic (static) | ~4.9 KB out + ~4.9 KB in | 17+6 scratch stores, 17+6 fills |
| sub-group reductions | 16 (8 max + 8 sum) | each a log2(16)-step shuffle tree |
| exp calls | 40 (32 P + 8 corr) | 80 `math.exp` after lowering |
| O rescale | 128 SIMD16 `mul` | every tile, even when the max did not move |
| causal mask | 8 cmp + 8 sel per S atom, 4 atoms | every tile, at every depth |

### 1.2 The instruction mix (static, main loop, the box's ocloc on 2026-09-27)

Arms are variants of the production source with one lever each (the appendix says how
to rebuild them; production is unchanged). Loop = the backward-jump loop
holding the DPAS; counts are static (both sides of a branch count).

| arm | spill (GRF / bytes) | loop instr | non-DPAS per DPAS | `math.exp` | `mad` | `mul` | `sel`+`cmp` | int (shr/and/add3) | scratch msgs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| production (`exp`, KT64 R16 H6 Q0) | 77 / 4928 | 1639 | 11.8 | 80 | 88 | 280 | 329 | 136 | 46 |
| + Q streamed per dim-pair (QS) | 82 / 5248 | 1650 | 11.9 | 80 | 88 | 280 | 329 | 136 | 52 |
| + QS + diagonal-only mask (DM) | 81 / 5184 | 1753 | 12.7 | 80 | 88 | 312 | 363 | 137 | 48 |
| + QS + DM + `exp2`, scale folded (E2) | **0** | **1137** | 7.9 | 33 | 8 | 192 | 171 | 128 | **0** |
| same with `native_exp2` | 0 | 1137 | 7.9 | 33 | 8 | 192 | 171 | 128 | 0 |
| + lazy rescale (LR) | 0 | 1212 | 8.5 | 34 | 8 | 192 | 194 | 128 | 0 |
| + hardware bf16 convert (CVT) | 0 | **1084** | 7.5 | 34 | 8 | 192 | 194 | **0** | 0 |
| KT32, QS only | 58 / 3712 | 1051 (64 DPAS) | 15.4 | 48 | 56 | 216 | 201 | 72 | 39 |
| KT32, all of the above | 0 | 712 (64 DPAS) | 10.1 | 18 | 8 | 160 | 114 | 0 | 0 |
| KT128, QS only | 101 / 6464 | 2812 (256 DPAS) | 10.0 | 144 | 152 | 408 | 585 | 264 | 62 |
| KT128, all of the above | 115 / 7360 | 2361 (256 DPAS) | 8.2 | 66 | 8 | 256 | 355 | 13 | 206 |

Readings:

- **The spill is `exp`'s, not Q's.** Streaming Q (QS) changes nothing (the compiler
  schedules the Q loads the same way); replacing `exp` removes all 77 spilled GRF. So
  spec 6's "Q0 beats Q1" finding was a spill finding, and the Q0/Q1 axis is worth
  re-sweeping once `exp` is gone.
- **`exp` costs 2 `math.exp` + ~6 ALU per value** here; `exp2` costs one `math.exp`.
  IGC's vISA documents `EXP` as base-2 (`intel-graphics-compiler/documentation/visa/
  instructions/EXP.md`), and the dump shows `exp2`, `native_exp2` identical.
- **The dynamic count is lower than the static one once LR and DM are in**: on a tile that
  is not on the diagonal and where no row max grew by more than 2^8, the mask path
  (~50 instr) and the rescale path (128 `mul` + the `corr` exps) do not run, so the
  steady-state loop is about **800 instructions per 128 DPAS (estimated)**, against 1639
  plus 46 scratch messages today: roughly half.
- **KT128 does not fit** (spills even with every lever); **KT32 with every lever fits
  with room** (712 per 64 DPAS; more instructions per DPAS than KT64 because the per-tile
  reductions amortise over half the keys) and is the natural base for a larger per-SG M.

### 1.3 Occupancy and scheduling (derived)

- 256 GRF gives 4 hardware threads per XVE, 32 per Xe-core. A work-group is
  6 heads x 16 rows / 8 = **12 sub-groups**, so an Xe-core holds 2 WGs = **24 of 32
  thread slots (75 %)**, 3 threads per XVE. With 6 heads per WG and 8 rows per SG, the WG
  size is always a multiple of 6 SGs, so 100 % is reachable only at 6 SGs (5 WGs = 30
  slots, 94 %) — `RPW=8`, never swept in spec 6.
- Softmax/ALU work and DPAS overlap only across threads (Xe2 co-issues from different
  threads to the FPU/EM/XMX pipes). At today's mix a thread needs ~1500 ALU/send issue
  slots per 2048 XMX cycles. That looked like the bottleneck on paper, but the measured
  arms say otherwise: once the spill is gone, removing another ~300 instructions per tile
  (DM, LR, CVT, DS) buys nothing, while one more thread per XVE (R8) buys 1.14-1.27x. The
  kernel is **latency / memory bound with too few threads to hide it**, not issue bound.
- **Causal tail.** Grid (C/16, 4): 512 WGs at C = 2048, 64 resident (2 per Xe-core), so 8
  waves. WG x has work proportional to pos/16 + x, and x is dispatched lowest first, so
  the heaviest WGs start last. For the first chunk (pos 0) the tail can reach one maximal
  WG, ~25 % of the ideal span (derived: max job 128 units against 32768/64 = 512); for
  pos 2048 about 8-17 %; at depth it vanishes. sycl-tla reverses the q-block order for
  exactly this (`applications/flash_attention_v2/kernel/xe_tile_scheduler.hpp`:
  `make_coord(params.grid.y - 1 - BlockIdxY(), ...)`). Our lever: `x' = ngroups - 1 - x`.
- **No cooperative prefetch, no barrier.** The 12 SGs of a WG read the same K/V tile but
  drift freely, and nothing prefetches tile t+1. vLLM and sycl-tla both issue a
  cooperative 2D prefetch of the next K tile and the current V tile, split across the
  WG's SGs (`make_block_2d_prefetch<SGPerWG>`), and wrap each iteration in a split
  work-group barrier (`barrier_arrive` / `barrier_wait`) to keep the SGs in lockstep so
  the loads hit L1. Our GEMM already does both (`pf_gemm.cl`: 2-deep
  `intel_sub_group_2d_block_prefetch_16b_8r16x2c` + `intel_work_group_barrier_arrive`),
  and `pf_flash_attn.cl` enables `cl_intel_split_work_group_barrier` without using it.

### 1.4 The operand-bandwidth ceiling (derived; the one number we lack)

A `dpas.8x8` bf16 reads an A operand (8 x 16 bf16 = 256 B) and a B operand
(16 x 16 = 512 B) and takes 16 XVE cycles; 8 XVEs per Xe-core.

| kernel | SG tile | B reuse | A reuse | operand bytes / DPAS | at 100 % DPAS, per Xe-core |
|---|---|---:|---:|---:|---:|
| `pf_gemm` (161.7 TFLOP/s) | 32 x 64 | 4 | 4 | 192 B | 96 B/clk |
| `pf_flash_attn` QK | 8 rows x 64 keys | 1 | 4 | 576 B | 288 B/clk |
| `pf_flash_attn` PV | 8 rows x 256 d | 1 | 4 (P) | 576 B | 288 B/clk |
| split-d, 16 rows / SG | 16 x 32 keys, 16 x 128 d | 2 | 2 | 384 B | 192 B/clk |
| split-d, 32 rows / SG | 32 x 16 keys, 32 x 64 d | 4 | 1-4 | ~200-320 B | ~100-160 B/clk |

At our 28 TFLOP/s the kernel draws ~44 B/clk per Xe-core; vLLM at 52 TFLOP/s ~82 B/clk;
the GEMM ~85 B/clk. **Intel publishes no Xe2 L1 load bandwidth we could find**, so the
ceiling of an 8-rows-per-SG kernel is unknown: if the L1/LSC path delivers ~128 B/clk per
Xe-core it is ~44 % of peak (~81 TFLOP/s); if 256 B/clk, ~89 %. Probe P0 below measures
it directly and decides whether split-d is needed to go past vLLM.

## 2. What the best Intel implementations do

Sources read from the box's copies (`~/vllm-xpu-kernels`, the build vLLM ran with;
`~/sycl-tla` at our pin 91e5bd7) and upstream (oneDNN, IPEX, Triton-XPU, llama.cpp).

| | vLLM-XPU `chunk_prefill` (63.1 ms here) | sycl-tla `xe_fmha_fwd` | oneDNN micro-SDPA | IPEX XeTLA | ours |
|---|---|---|---|---|---|
| hd256 tile | WG 256 q rows x 1 head, 32 SGs x 8 rows, KV tile 32, d-chunk 32 | no hd256 config (64-192); hd128: 16 SGs x 16 rows | xe2/256: WG 128 keys x 128 q, 32 SGs; S tile 16 keys x 32 q per SG; **O split on d** (8 SGs x 32 d) | 64x128x256 etc., **O split on d** (kSgHm 32-128) | WG 16 rows x 6 heads, 12 SGs x 8 rows, KV tile 64 |
| O per SG | 8 x 256 fp32 (128 GRF) | 16 x 128 at hd128 | 32 q x 32 d (~64 GRF) | 16 x 32..128 d | 8 x 256 (128 GRF) |
| GQA | one q head per WG; kv shared via L1/L2 | same | per head | per head | 6 q heads per WG (spec 6's biggest lever) |
| K/V source | global 2D block loads, no SLM | same | cooperative prefetch; Q and P in SLM | SLM | global 2D block loads, no SLM |
| prefetch / sync | coop 2D prefetch next K, current V; split WG barrier per tile | same | coop prefetch `LSC_LDCC_L1C_L3C`; split barriers | barriers | none |
| exp | `native::exp2(scale*s - m)`, scale*log2e folded | same | `native_vexp2(x*scale*log2e)` | exp2 | `exp(s*scale - m)` |
| row sum | reduced per tile | **deferred**: per-lane partials, one reduce at the end | SLM | | reduced per tile |
| O rescale | per V sub-tile, fused before the PV DPAS; skipped on the first tile | same | per tile | per tile | every tile, all 16 vectors |
| causal mask | only tiles with K >= first causal tile | only the last tile of each SG | per tile, predicated | | every element of every tile |
| q-block order | ascending | **reversed** (heaviest first) | | | ascending |
| bf16 P | `reorder` (hardware cvt) | paired `cvt_f32x2_to_bf16x2` | | | integer RNE (4 ops) |

Paths: vLLM `csrc/xpu/attn/xe_2/fmha_utils.hpp` (`chunk_policy_head256`:
`ShapeQK<_256,_32,_32>`, `ShapePV<_256,_32,_32>`, `ShapeOut<_256,_256>`,
`SubgroupLayoutQK<_32,_1,_1>`), `xe_2/collective/chunk_prefill_mainloop.hpp` (main loop,
`softmax()`), `xe_2/kernel/chunk_prefill_kernel.hpp` (causal trip counts); sycl-tla
`applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp`,
`kernel/xe_tile_scheduler.hpp`, `examples/06_bmg_flash_attention/`; oneDNN
`src/gpu/intel/sdpa/configs.cpp` (`{xe2, 256} -> {16, 32, 32, 32, 8, 4, 8, 4}`) and
`micro.cl`; IPEX `csrc/gpu/aten/operators/xetla/kernels/SDP/fmha_forward_policy.h`;
Triton-XPU `benchmarks/triton_kernels_benchmark/flash_attention_benchmark.py`
(BLOCK_M 128/256, BLOCK_N 32/64, `grf_mode 256`, exp2 with folded log2e, causal split into
an unmasked off-band loop and a masked band); Codeplay, "Improving Triton FlashAttention
performance on Intel GPU" (2025-09-02: sinking Q/K loads into the loop to cut liveness,
+5-10 % on PVC); FlashAttention-4 (tridao.me/blog/2026/flash4, Dao-AILab PR #2907:
conditional rescale with threshold 8 in log2 units, final normalisation with the true
statistics); llama.cpp SYCL has no XMX flash kernel of its own (oneDNN Graph SDPA or
GEMM). No project publishes hd256 prefill TFLOP/s on Xe2.

What this says about vLLM's 2x: **per-SG it is our shape** (8 rows x 256 d O, B reuse 1,
no SLM). It wins on the instruction side (native exp2, masks only on the diagonal,
hardware bf16, rescale fused and skipped on the first tile, no spill at KV tile 32), on
100 % thread occupancy (32 SGs, one WG per Xe-core) and on L1 locality (cooperative
prefetch + a split barrier per tile). It loses the GQA sharing we have (6 heads per WG),
so our kernel with its instruction mix fixed should land at or past vLLM. The measured
arms confirm half of this: exp2 and occupancy were worth 1.42x; the instruction-count
cleanups were not. The rest of vLLM's lead is plausibly its cooperative prefetch and
per-tile barrier (L1 locality), which we have not tried yet.

## 3. head_dim 256: the O accumulator

- **Whole O per SG, 8 rows** (vLLM, ours): 128 GRF, B reuse 1. Fits 256 GRF only with a
  lean softmax (KV tile 32 in vLLM; ours at 64 fits once `exp` is gone).
- **O split on d across SGs, S/P shared through SLM** (oneDNN micro-SDPA, IPEX XeTLA):
  the SGs of a row block each own a d-slice of O; QK^T is split on keys (each SG computes
  S for its key slice over the full d), the row max/sum are combined in SLM, and P is
  written to SLM (VNNI-packed) for every SG's PV. Lets an SG hold 16-32 rows, which is the
  only way to B-reuse >= 2 at head 256.
- **O split on d across work-groups** (sycl-tla's grid x = head_size_vo / ShapeOut_v):
  recomputes QK^T per d-slice; 1.5x the FLOPs for 2 slices. Not attractive here.
- **fp32 O in SLM**: nobody does it for prefill (SLM bandwidth equals L1's; it only moves
  the traffic).

## 4. Ranked levers

Expected gains are **estimates** against today's kernel on pp4096's attention phase
(116.3 ms) unless a measured row above says otherwise; they compound, not add.

| # | change | mechanism | expected gain | risk | how to probe |
|---|---|---|---|---|---|
| 1 | **`exp` -> `exp2` with `ATTN_SCALE*log2e` folded** | removes `exp`'s range-split sequence; the 77-GRF spill (4.9 KB scratch out + in per tile) disappears; loop 1639 -> 1137 instr | **measured 1.17-1.23x** | low: K1 cosines bit-identical; needs the numerics ruling above | done (§0); promote with 2 |
| 2 | **RPW 16 -> 8: 6 sub-groups per WG** (grid `(ceil(C/8), 4, 1)`) | 30 of 32 thread slots per Xe-core instead of 24; one more thread per XVE to hide load latency; finer causal granularity | **measured 1.14-1.27x alone, 1.42x with 1** | low: same text, one define and the grid in `attn.cc`; K1 passes | done (§0); confirm in-engine pp4096 / pp65536 |
| 3 | **cooperative 2D prefetch of the next K tile (and current V) + split WG barrier per tile** | what vLLM, sycl-tla and our own `pf_gemm` do; keeps the 6 SGs (which at R8 share one K/V stream and identical trip counts, so the barrier is safe) in lockstep on L1, and hides the L2/DRAM latency the R8 result shows we are bound by | est. 1.1-1.3x | medium: barrier + prefetch distance tuning; no numerics change | arms on top of #1+#2: prefetch K only / K+V / +barrier; distance 1 and 2 |
| 4 | **split-d: 16 query rows per SG, O split on d over an SG pair, S split on keys, P and row stats exchanged through SLM** | B-operand reuse 2 instead of 1 (operand bytes per DPAS 576 -> 384); the only way past the M=8 ceiling at head 256 (oneDNN, IPEX XeTLA do this) | est. 1.3-1.8x beyond #3 **if** P0 shows the M=8 ceiling is near | high: new kernel, SLM + barriers, register fit at KT32 (~220 GRF est.) | P0 first; then a prototype arm at KT32, K1 + timing vs the #3 winner |
| 5 | **larger K messages** (`transpose_32b_32r8x1c`, 32 keys per send) | halves K send count (64 -> 32 per tile); fewer LSC messages in flight per DPAS | est. 1.0-1.1x | low-medium: the register layout of the 32-row transposed read must be pinned first (our guess failed) | a layout unit probe (write iota, read back), then one arm |
| - | measured neutral, deprioritise: diagonal-only mask, conditional rescale (FA-4), hardware bf16 cvt, deferred row sums, reversed row order, KV tile 32 vs 64 | the kernel is no longer issue-bound | 0.97-1.01x measured | - | keep DM and CVT as free cleanups only if a later arm becomes issue-bound again |


## 5. Probe plan

One lever at a time, each arm in `probe_flash_attn` style (same harness, fp64 K1 bar
cos >= 0.99999 on the five cases of §0, pad rows finite, composed control, 20 warm-up +
11 interleaved paired rounds, device 0, `flock`, detached), at pos 0 / 2048 / 30720 and
one deep row (63488, C 2048, needs max_len 65536 in the harness).

- **P0 - operand-bandwidth ceiling (1 small probe, decides #4).** A variant of
  `probe_dpas_rates`: a loop of `dpas.8x8` whose B operand comes from 2D block loads of an
  L1-resident tile (a 32-64 KB window, re-read), at B reuse 1, 2 and 4, with the
  transposed-32b and the VNNI-transform read, and with / without a 2D prefetch. Output:
  TFLOP/s per reuse. If reuse 1 already reaches >= ~60 % of peak, skip split-d and spend
  the effort on #3; if it caps near 30-45 %, split-d is the road past vLLM.
- **P1 - promote #1 + #2** (pfa_KT64_R8_H6 with `exp2`): the engine's pp4096 attention
  phase (`B70_PREFILL_PROFILE=1`, the `attn_flash` row), pp4096 / pp65536 / pp131072 t/s,
  K2-K4 and spec 6 K3a's flash-vs-oracle check. Expected ~82 ms and ~1.42x at depth.
- **P2 - prefetch and barrier (#3)** on the P1 kernel: arms (a) next-K prefetch spread
  over the 6 SGs, (b) + current-V prefetch, (c) + `intel_work_group_barrier_arrive/wait`
  around each tile, (d) prefetch distance 2. Also re-sweep RPW 8 against the 12-SG shape
  with (c), since lockstep changes the L1 picture.
- **P3 - split-d prototype (#4)**, only if P0 says so: KT32, 16 rows per SG, 2 SGs per
  row block owning d 0-127 / 128-255, S computed on key halves, row max/sum and the bf16
  P half (1 KB) exchanged through SLM per tile. First a static ocloc pass (must be
  spill-free), then K1, then timing against the P2 winner.
- **P4 - leftovers**: the 32-row transposed K read (#5) after a layout unit probe; Q held
  in registers (Q1) re-swept now that `exp`'s temporaries no longer spill (static check
  first: 128 O + 64 Q + S/P/V at KT32 is ~248 GRF, borderline).

Stopping rule: an arm that fails K1 is recorded and not timed (as §0 did for the 32-row
K read); a lever that measures within +-2 % of its control on all three depths is
recorded as neutral and not promoted.


## Appendix: reproduction

- Static arms: `ocloc compile -file <arm>.cl -device bmg-g31 -options "-cl-std=CL3.0
  -cl-fp32-correctly-rounded-divide-sqrt -D... -cl-intel-256-GRF-per-thread"` with
  `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>`; the loop is the backward `jmpi`
  holding the DPAS in `*_simd16_entry_0001.asm`.
- The lever arms of §0: `tools/probe/pfa_levers_research.cl` (committed, not built).
  To rerun, copy it over `tools/probe/probe_flash_attn.cl` and put the arm list
  (`"64 16 6 0" "64 16 6 2" "64 8 6 0" "64 8 6 2" "64 8 6 47" "64 8 6 63" "32 8 6 2"
  "32 8 6 63"` ...) into the PFA foreach of `tools/probe/CMakeLists.txt` and `kArms` of
  `probe_flash_attn.cc`; `probe_flash_attn <pos> 2048 --arms all --time`.
- The static table of §1.2 used intermediate variants (a Q-streaming path, KT128)
  compiled with the same flags; they are reproducible from the lever file plus the QS
  path it still carries (`#define QS 1`).
