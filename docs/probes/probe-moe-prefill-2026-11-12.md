# MoE prefill probe, 2026-11-12: the expert GEMM straight from int4, and the multi-work-group sort

Plan `docs/superpowers/plans/2026-11-12-ornith-moe-prefill.md`, Task 1. Probe `tools/probe/probe_moe_prefill`
(`.cc` + `.cl`), box row 83 (`r83.routes`, `r83.probe`). Card 0, box state not checked idle (another agent's card
jobs ran on card 1 at times): **busy box**. Every time is an L0 timestamp span around one arm's whole walk over
every layer-chunk of a real route dump, after a warm-up, arms interleaved in rotating order, median of 5 rounds
(3 for the rounds marked so), divided by the layer-chunks.

**Verdict: GO for Tasks 2-4.** The best bitwise fused arm does a layer-chunk's weights + grouped GEMMs in
**2.74 ms at C = 2048** against the pre-registered break-even 6.7 ms and target 4.2 ms (the profile's 8.35 ms; this
harness's own control measures 10.66). The sort goes from 1.50 to **0.048 ms** (gate 0.3). Flash-Next: 27.1 ->
5.65 ms a layer-chunk at C = 2048, sort 2.94 -> 0.078 ms.

## What ran

- **Routes**: a real prefill's route rows of every layer. Ornith: `ornith_prefill_test --dump-routes`, l0, the
  first 4096 ids of `long32k` in two chunks of 2048 (40 layers). Flash-Next: `qwen4exp_prefill_test
  dump-routes:`, Intel's checkpoint at `layers:18` with the int8 head, the first 4096 ids of `q4exp_4k`. C = 2048
  walks 2 chunks a layer (80 / 36 layer-chunks), C = 4096 one (40 / 18). (Plan deviation: no separate
  `--prefill-chunk 1024` dump; C = 4096 is the chunk Task 6 needs.)
- **Weights / activations**: random int4 layout-1 blocks (two copies alternated by layer-chunk), random bf16 rows;
  the routes - so the tile tables, item tables and expert row counts - are the dumps'.
- **Control**: the production walk, `pf_moe_dequant_gu` x 2 batches + `pf_moe_gemm` SiLU, `pf_moe_dequant_dn` +
  `pf_moe_gemm` (Flash-Next: `q4_pf_dequant_*`).
- **Gates before any time** (all held, on 4 layer-chunks a C and model): the mw sort's header, tile table,
  `row_tok`, `pair_row` and work items equal the host twin (`pf_moe_ref::sort`) and, at C = 2048, the
  one-work-group sort; every bf16 arm's h and y equal the control's over every used row as raw 32-bit words; every
  W4A8 arm's fp32 result (its EPI 2 binary) equals an integer-exact host oracle on two tiles' rows x every column.

## Arms

`F<form>.a<NA>b<NB>m<WGM>n<WGN>.c<CVT>x<XSG>o<OPT>`: a sub-group is 8 NA rows x 16 NB columns, a work-group WGM x
WGN sub-groups, a work item MI = 8 NA WGM rows of one expert.

| form | what |
|---|---|
| F1 | bf16, each sub-group converts its own B columns in registers (pf_gemm_q4.cl's layout-1 read) |
| F2 | bf16, the work-group's 64-k B slab converted once into double-buffered SLM, shared by its row sub-groups |
| F3 | W4A8: int8 rows x the int4 block as the native s4 DPAS operand (the nibble xor 8 is the whole conversion, exact), int32 per 64-k group, fp32 rescale per group: x0 the g64 weight scale per group and the per-row scale at the end; x1 weight scale x the row's per-group scale per group |
| CVT | the B value rne((q - 8) scale): 0 pf_gemm_q4's statement for statement; 1 the same float, hardware f32 -> bf16; 2 magic exponent fma(16 + q, s, -24 s) then the hardware conversion; 3 inline vISA (bfe, int -> f32, one mixed-mode bf16 multiply) - **not bitwise, dropped**; 9 DIAGNOSTIC: no conversion |
| OPT | 1 n-blocks on grid dimension 0 (an item's work-groups together: its A once into L2); 2 SLM block IO (F2); 4 DIAGNOSTIC: no DPAS |

## Ornith (E 256, top 8, H 2048, I 512), ms a layer-chunk

| arm | C 2048 gu | dn | sum | C 4096 gu | dn | sum | bitwise |
|---|---:|---:|---:|---:|---:|---:|:--:|
| control | 6.950 | 3.708 | **10.66** | 7.792 | 4.407 | **12.20** | - |
| F1.a4b4m1n4.c0x0o0 (the plan's F1) | 3.189 | 1.657 | 4.85 | 5.741 | 2.942 | 8.68 | yes |
| F1.a4b4m1n4.c2x0o1 | 2.239 | 1.185 | 3.42 | 3.828 | 2.058 | 5.89 | yes |
| F2.a4b4m4n4.c2x0o3 (MI 128) | 2.138 | 1.073 | 3.21 | 3.126 | 1.639 | 4.77 | yes |
| **F1.a8b2m1n8.c2x0o1 (MI 64) - chosen** | **1.770** | **0.966** | **2.74** | **2.807** | **1.541** | **4.35** | yes |
| F1.a8b2m1n8.c9x0o1 DIAGNOSTIC (no conversion) | 1.072 | 0.647 | 1.72 | 1.692 | 1.027 | 2.72 | - |
| F3.a4b4m1n4.c0x0o1 (W4A8, per-row x) | 1.106 | 0.656 | 1.76 | 1.917 | 1.127 | 3.04 | oracle |
| F3.a4b2m4n8.c0x0o1 | 1.137 | 0.664 | 1.80 | 1.661 | 1.049 | 2.71 | oracle |
| F3.a4b2m4n8.c0x1o1 (W4A8, per-row-and-group x) | 1.488 | 0.782 | 2.27 | 2.212 | 1.231 | 3.44 | oracle |
| sort: pf_moe_sort -> pf_moe_sort_mw | 1.495 -> **0.048** | | | - -> 0.058 | | | yes |

TFLOP/s (useful rows only, routed pairs + the shared expert's C): the chosen arm 43.7 / 40.0 (C 2048) and 55.1 /
50.2 (C 4096) against the control's 11 / 10 and 20 / 18 (the profile's 33.5 was the GEMM alone, without its
weight pass); without the conversion it would be 72 / 60 and 91 / 75.

The W4A8 arms' activation quantiser (`pmp_quant`, a probe kernel over the GATHERED rows, one work-group a row) is
timed separately: 1.0-2.0 ms a layer-chunk per-row, 3.8-7.0 per-group. It is not a design: production quantises
the C token rows once before the gather (x1: one scale per 64 k), ~9x fewer rows; it is left out of the sums.

Rounds that led here (C = 4096, 3 rounds each): CVT 2 is 28 % faster than CVT 0 (the software rne); the grid
order (OPT 1) 4-11 %; F2's SLM block IO (OPT 2) 6 %; MI 64 or 128 beats 32 (half-empty 64-row items at C 2048
still win: half the conversion per DPAS); MI 256 loses. 64-row sub-groups first failed the bitwise gate (rows past
a 32-row item belong to the next item - fixed by ending the store surface at the item's last row; the A read's row
offset also went into the base).

## Flash-Next (E 512, top 10, H 2560, I 640), ms a layer-chunk (3 rounds)

| arm | C 2048 gu | dn | sum | C 4096 gu | dn | sum |
|---|---:|---:|---:|---:|---:|---:|
| control | 17.849 | 9.278 | **27.13** | 20.482 | 11.067 | **31.55** |
| F1.a4b4m1n4.c2x0o1 | 4.438 | 2.340 | 6.78 | 7.578 | 4.001 | 11.58 |
| F2.a4b4m4n4.c2x0o3 | 4.835 | 2.322 | 7.16 | 6.830 | 3.387 | 10.22 |
| **F1.a8b2m1n8.c2x0o1** | **3.699** | **1.950** | **5.65** | **5.673** | **3.033** | **8.71** |
| F3.a4b2m4n8.c0x0o1 | 2.663 | 1.380 | 4.04 | 3.818 | 2.086 | 5.90 |
| sort | 2.939 -> **0.078** | | | - -> 0.090 | | |

All bitwise / oracle-exact as Ornith's.

## Where the time goes (the conversion is not hidden)

The diagnostic arms split the chosen arm: at C = 4096 the DPAS walk without conversion is 2.72 ms and the
conversion without DPAS (F1 c2 o5, a4b4) 1.52 for gate||up alone - the sum, not the max: the conversion's ALU does
not overlap the XMX pipe (the 2026-09-23 W4A8 rescale finding, again). The loop's ISA (shader dump, a8b2 c2): 64
`dpas.8x8` against ~700 ALU instructions a 64-k group (shl / shr / bfn / mad / the bf16 `mov` / packing `or`),
about 5.5 a converted value; the vISA conversion cut that by 20 % on paper but was not bitwise and was dropped.
What is left toward the plan's 40-50 % of bf16 peak (Task 4): software-pipelining the conversion one k-step ahead
of its DPAS, fewer instructions a value, and a 128-row item at C = 4096 with SLM sharing (F2) once its barrier and
SLM costs come down (F2's conversion-only time, 3.4 ms, is all overhead).

## Numerics of the W4A8 forms (Ornith, real activations, against f32)

`tools/probe/moe_w4a8_accuracy.py` (CPU, the f32 reference's own MoE inputs captured by hook): prose prompt,
layers 1, 8, 15, 22, 29, 36, 39; the MoE block output, gate||up's h and down alone, each form against the f32 form
of the same weights. Bar of this tensor study: worst row cosine 0.999 (the engine gate is the f32 judge, Task 5).

| form | block relL2 | block worst cos | x l0 | h relL2 | down relL2 |
|---|---:|---:|---:|---:|---:|
| l0 (bf16, the engine) | 0.333 % | 0.999987 | 1.0 | 0.335 % | 0.165 % |
| i8 (today's l0-int8: rotated int8 gate||up, bf16 down) | 1.024 % | 0.999670 | 3.0 | 1.222 % | 0.165 % |
| W4A8 per-row x (x0) | 3.372 % | 0.978764 | 10.0 | 2.485 % | 2.544 % |
| W4A8 per-row-and-64-group x (x1) | 1.458 % | 0.999083 | 4.3 | 1.006 % | 1.176 % |
| **W4A8 x1 gate||up + bf16 down** | **0.902 %** | **0.999146** | **2.7** | 1.006 % | 0.165 % |
| W4A8 x0 gate||up + bf16 down | 2.187 % | 0.978935 | 6.5 | | |
| rotated RTN checkpoint, per-row x | 12.43 % | 0.964838 | 37.1 | 10.44 % | 9.64 % |
| rotated RTN checkpoint, per-group x | 12.41 % | 0.965184 | 37.0 | | |

- Per-row W4A8 fails on both GEMMs, as on Qwen3.8 on 2026-09-23.
- **Per-row-and-group activation scales with down kept bf16 is the one W4A8 form at or below today's l0-int8
  (0.90 % vs 1.02 % relL2, x2.7 vs x3.0 of l0)**: the candidate for Task 5, to be judged end to end by the f32
  judge.
- The rotated arm re-quantises the CHECKPOINT's int4 weights after the rotation (RTN g64 of H^T W): a second int4
  rounding on top of the first, so its 12 % is that double quantisation measured against the checkpoint - it
  cannot price a checkpoint re-quantised with the rotation from the bf16 original (that one would be judged
  against the bf16 model, not this one). Recorded, not a basis for a decision.

## Files

`tools/probe/probe_moe_prefill.{cc,cl}`, `tools/probe/moe_w4a8_accuracy.py`, the dump modes in
`tests/golden/ornith_prefill_test.cc` and `tests/runtime/qwen4exp_prefill_test.cc`, `src/kernels/prefill/
pf_moe_sort_mw.cl`. Logs on the box: `~/b70/logs/ornith-moe-prefill/` (probe1, probe_c4096, probe_r2c4096,
probe_r3*, probe_r4, probe_r5, probe_final, probe_fn, numerics_prose).
