# MoE prefill probe, 2026-11-12: the expert GEMM straight from int4, and the multi-work-group sort

Plan `../superpowers/plans/2026-11-12-moe-model-prefill-phase1.md`, Task 1. Probe `tools/probe/probe_moe_prefill`
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

## Task 8: W4A8 on l0-int8, the gates and the record

**The switch-over (operator 2026-11-12):** W4A8 - per-token x per-64-group int8 activations, int8 x int4 DPAS for
gate||up, bf16 down - is `l0-int8`'s only MoE expert path; the rotated int8 gate||up, the old walk, the weight batch
and `B70_PF_MOE` are gone (no env variables). The `l0-int8` MoE gates take the plain f32 judge under the near-miss
rule in `prefill_gate_test`: a determined row the engine misses passes when the f32 margin is under 8 bf16 ulps AND
either the bf16 twin misses it too or the engine's logits cosine vs f32 is >= the twin's - 1e-5 (both relaxations:
operator 2026-11-12). Every row that passes only by it is printed.

**Gates (branch on main 7cfbf058, card 0):** `pf_moe_test` PASS (W4A8 gate||up exact against the host's integer
sums at C 2048 and 37); `prefill_gate_ornith_l0` 90/90 determined, `_c16` 89/90 + 1 near-tie; `l0-int8` 89/90 -
cjk 15 passes only by the rule (f32 3.0 ulp, cosine 0.999465 vs the twin's 0.999467); `l0-int8_c16` 89/90 - prose
18 passes only by the rule (f32 7.2 ulp, the twin picks the same 32); the prefill splits, `ornith_prefill_l0` /
`l0-int8`, `ornith_decode_test`, Flash-Next `qwen4exp_pf_kernels` / synth ours / intel PASS. Ornith `l0` routes and
logits and Flash-Next synthetic routes (ours, Intel's) are bitwise main's. K2, Kolibri and Bolt: no file of theirs
or kernel they build changed.

**The record** (main 7cfbf058 vs the branch, `b70-decode --bench --max-len auto`, card 0, median of 3, interleaved
main / branch; card 1 held by another agent's jobs and the CPU f32-references job running, load ~30):

| prompt pass | main 4k | branch 4k | main 16k | branch 16k |
|---|---:|---:|---:|---:|
| `l0` (bf16) | 540.5 ms, 7,578 t/s | **506.9 ms, 8,080 t/s** | 6,509 t/s | **6,800 t/s** |
| `l0-int8` | 750.2 ms, 5,460 t/s | **480.9 ms, 8,518 t/s** | 4,869 t/s | **7,059 t/s** |

Against the plan's start (2026-11-11: 3,362 / 3,937 t/s at 4k): `l0` 2.40x at 4k, 2.09x at 16k (gate 1.7x / 1.5x);
`l0-int8` faster than `l0` at both lengths; decode unchanged (tg 32 medians 129-135 t/s on both builds, noise of the
busy box). The branch plans 4096-row chunks (main 2048); prefill scratch 0.954 GB vs 1.040.

**Per phase at 4k** (`B70_PREFILL_PROFILE=1`, branch, ms) toward the operator's targets (`l0-int8` >= 11,000 t/s =
370 ms, `l0` >= 9,500 = 430 ms):

| phase | `l0` | `l0-int8` |
|---|---:|---:|
| MoE: route / sort / gather / expert GEMMs / combine | 27.5 / 1.8 / 17.4 / 169.8 / 15.3 = **231.8** | 27.0 / 2.0 / 12.0 / 125.2 / 15.3 = **181.5** |
| dense GEMMs (+ conversion or int8 quantisers) | 91.1 | 74.4 |
| GDN (projections, conv, solve, wu, scan, head) | 94.5 | 128.8 |
| attention | 47.8 | 46.3 |
| norms | 16.8 | 16.7 |
| sum / measured | 482 / 507 | 448 / 481 |

Left: `l0-int8` ~111 ms, `l0` ~77 ms. The expert GEMMs are the largest single phase (125 / 170 ms); GDN on
`l0-int8` (129 ms, the four-launch intra-chunk path) is the next.

**Margin table** (the evidence for the switch-over, chunk 2048, Ornith's 3 gate prompts, every determined row: the
f32 judge's margin of its top token over the runner-up and each form's margin of that token, bf16 ulps; negative =
another token ahead; "old l0-int8" is the rotated int8 form W4A8 replaced, i8bf16 = bf16 experts under int8 dense):

| prompt | row | f32 margin (ulp) | l0 | old l0-int8 (rotated int8) | W4A8 | i8bf16 | miss |
|---|---:|---:|---:|---:|---:|---:|---|
| prose | 1 | 16.0 | 15.8 | 19.1 | 16.7 | 17.5 |  |
| prose | 2 | 5.2 | 6.5 | 2.5 | 5.2 | 7.0 |  |
| prose | 3 | 3.8 | 3.4 | 3.8 | 4.6 | 3.9 |  |
| prose | 4 | 5.2 | 4.7 | 4.4 | 5.7 | 4.7 |  |
| prose | 5 | 29.6 | 30.1 | 28.3 | 29.5 | 29.9 |  |
| prose | 6 | 5.4 | 4.8 | 4.4 | 4.8 | 4.4 |  |
| prose | 7 | 40.0 | 44.5 | 43.6 | 42.1 | 43.0 |  |
| prose | 8 | 2.9 | 0.2 | 0.7 | 1.9 | 1.6 |  |
| prose | 9 | 71.5 | 72.7 | 74.4 | 74.3 | 75.6 |  |
| prose | 12 | 34.7 | 35.9 | 35.6 | 33.9 | 34.5 |  |
| prose | 13 | 20.2 | 20.0 | 20.0 | 21.1 | 19.4 |  |
| prose | 14 | 7.3 | 8.9 | 7.0 | 9.1 | 6.6 |  |
| prose | 15 | 3.6 | 3.4 | 1.6 | 2.8 | 1.1 |  |
| prose | 17 | 8.6 | 6.8 | 6.5 | 6.5 | 7.7 |  |
| prose | 18 | 7.2 | 10.4 | 7.5 | 8.9 | -1.1 | i8bf16 |
| prose | 19 | 12.7 | 7.5 | 18.0 | 12.5 | 14.2 |  |
| prose | 20 | 17.4 | 12.2 | 16.2 | 19.1 | 10.2 |  |
| prose | 21 | 7.0 | 5.6 | 3.7 | 7.5 | 1.4 |  |
| prose | 22 | 3.4 | 3.2 | -0.3 | 2.2 | 2.5 | i8 |
| prose | 23 | 7.6 | 7.9 | 2.1 | 6.8 | 6.9 |  |
| prose | 24 | 1.7 | 2.8 | 3.5 | 4.1 | 1.8 |  |
| prose | 25 | 30.2 | 30.8 | 32.0 | 27.3 | 30.2 |  |
| prose | 26 | 48.2 | 42.6 | 49.9 | 48.1 | 49.4 |  |
| prose | 27 | 135.9 | 136.3 | 130.8 | 140.5 | 136.8 |  |
| prose | 28 | 33.3 | 29.4 | 27.4 | 29.0 | 30.9 |  |
| prose | 29 | 41.4 | 41.2 | 41.2 | 41.6 | 41.8 |  |
| prose | 30 | 69.8 | 69.4 | 68.8 | 61.8 | 68.3 |  |
| prose | 31 | 53.1 | 57.0 | 58.2 | 54.3 | 56.7 |  |
| code | 0 | 9.3 | 9.6 | 9.6 | 9.7 | 8.8 |  |
| code | 1 | 1.7 | 1.8 | 2.9 | 1.6 | 3.0 |  |
| code | 2 | 16.2 | 17.8 | 19.6 | 19.9 | 17.9 |  |
| code | 4 | 5.5 | 5.8 | 5.1 | 5.6 | 4.0 |  |
| code | 5 | 40.8 | 42.8 | 42.0 | 42.3 | 42.4 |  |
| code | 6 | 50.5 | 52.1 | 53.5 | 51.7 | 51.8 |  |
| code | 7 | 94.7 | 95.0 | 95.0 | 94.9 | 96.6 |  |
| code | 8 | 141.7 | 144.7 | 143.9 | 143.2 | 144.6 |  |
| code | 9 | 51.9 | 53.1 | 52.1 | 49.4 | 51.1 |  |
| code | 10 | 119.6 | 118.2 | 116.8 | 117.2 | 119.5 |  |
| code | 11 | 131.8 | 132.9 | 133.3 | 132.8 | 133.1 |  |
| code | 12 | 11.9 | 13.1 | 15.1 | 13.5 | 13.1 |  |
| code | 13 | 39.8 | 38.9 | 44.0 | 42.0 | 40.8 |  |
| code | 14 | 40.6 | 42.5 | 43.4 | 42.9 | 42.5 |  |
| code | 15 | 138.3 | 142.0 | 142.0 | 142.5 | 141.5 |  |
| code | 16 | 107.1 | 112.0 | 111.6 | 111.1 | 108.7 |  |
| code | 17 | 120.3 | 119.8 | 127.8 | 119.6 | 118.7 |  |
| code | 18 | 169.8 | 166.4 | 165.7 | 163.7 | 172.4 |  |
| code | 19 | 32.5 | 31.2 | 31.5 | 33.0 | 35.6 |  |
| code | 20 | 73.1 | 76.5 | 74.1 | 74.7 | 74.6 |  |
| code | 21 | 120.0 | 121.2 | 120.2 | 119.0 | 124.2 |  |
| code | 22 | 135.3 | 138.4 | 141.2 | 137.2 | 135.5 |  |
| code | 23 | 6.6 | 7.3 | 7.6 | 7.5 | 6.7 |  |
| code | 24 | 134.1 | 136.3 | 139.3 | 137.0 | 133.7 |  |
| code | 25 | 124.8 | 126.8 | 125.3 | 124.6 | 127.0 |  |
| code | 26 | 146.1 | 144.2 | 143.3 | 142.5 | 144.0 |  |
| code | 27 | 171.1 | 161.9 | 171.1 | 171.9 | 170.7 |  |
| code | 28 | 156.7 | 156.9 | 154.0 | 150.5 | 159.6 |  |
| code | 29 | 189.4 | 189.8 | 189.2 | 190.9 | 190.9 |  |
| code | 30 | 160.1 | 167.0 | 161.6 | 164.1 | 163.4 |  |
| code | 31 | 157.1 | 160.5 | 158.6 | 158.1 | 156.3 |  |
| cjk | 0 | 46.6 | 45.1 | 46.3 | 45.8 | 45.1 |  |
| cjk | 1 | 8.0 | 7.7 | 10.0 | 7.9 | 9.1 |  |
| cjk | 2 | 12.6 | 12.3 | 13.7 | 14.4 | 13.0 |  |
| cjk | 3 | 10.7 | 11.7 | 9.3 | 10.8 | 8.7 |  |
| cjk | 4 | 17.3 | 16.1 | 17.4 | 17.5 | 17.1 |  |
| cjk | 5 | 40.1 | 40.9 | 39.2 | 38.9 | 39.7 |  |
| cjk | 6 | 30.3 | 29.8 | 30.4 | 30.1 | 30.9 |  |
| cjk | 7 | 18.4 | 17.8 | 18.7 | 20.4 | 16.7 |  |
| cjk | 8 | 5.8 | 8.7 | 6.4 | 5.5 | 7.4 |  |
| cjk | 9 | 2.5 | 3.6 | 1.4 | 3.1 | 2.4 |  |
| cjk | 10 | 24.4 | 23.4 | 25.5 | 26.0 | 25.9 |  |
| cjk | 11 | 4.1 | 4.5 | 6.3 | 5.8 | 2.0 |  |
| cjk | 12 | 8.8 | 9.8 | 9.9 | 9.1 | 8.8 |  |
| cjk | 13 | 16.2 | 19.1 | 20.7 | 14.4 | 18.7 |  |
| cjk | 14 | 4.1 | 9.0 | 8.3 | 6.2 | 6.7 |  |
| cjk | 15 | 3.0 | 3.0 | -0.5 | -2.5 | -0.4 | i8, w4a8, i8bf16 |
| cjk | 16 | 9.0 | 9.0 | 8.9 | 7.3 | 7.4 |  |
| cjk | 17 | 11.9 | 10.9 | 12.8 | 11.3 | 14.0 |  |
| cjk | 18 | 21.6 | 21.1 | 21.5 | 21.0 | 21.9 |  |
| cjk | 19 | 8.9 | 8.5 | 8.8 | 8.4 | 9.0 |  |
| cjk | 20 | 11.9 | 7.5 | 8.3 | 9.6 | 9.9 |  |
| cjk | 21 | 14.0 | 12.5 | 13.8 | 11.2 | 8.3 |  |
| cjk | 22 | 39.0 | 37.8 | 39.8 | 42.2 | 39.4 |  |
| cjk | 23 | 14.4 | 13.7 | 17.9 | 14.9 | 14.0 |  |
| cjk | 24 | 3.1 | 1.5 | 0.3 | 4.7 | 4.0 |  |
| cjk | 25 | 17.5 | 14.8 | 15.3 | 15.7 | 23.6 |  |
| cjk | 27 | 16.1 | 18.7 | 19.4 | 16.0 | 15.1 |  |
| cjk | 28 | 6.5 | 5.3 | 7.7 | 6.2 | 6.0 |  |
| cjk | 29 | 14.5 | 22.3 | 17.9 | 12.8 | 19.9 |  |
| cjk | 30 | 11.6 | 14.7 | 11.1 | 11.0 | 11.6 |  |
| cjk | 31 | 32.8 | 31.5 | 32.6 | 31.8 | 34.4 |  |

- f32 margin < 2 ulp: 2 rows; right: l0 2, i8 2, w4a8 2, i8bf16 2
- f32 margin < 4 ulp: 9 rows; right: l0 9, i8 7, w4a8 8, i8bf16 8
- f32 margin < 8 ulp: 23 rows; right: l0 23, i8 21, w4a8 22, i8bf16 21
- all determined: right l0 90, i8 88, w4a8 89, i8bf16 88
- l0 misses: 
- i8 misses: prose 22 (f32 3.4, engine -0.3); cjk 15 (f32 3.0, engine -0.5)
- w4a8 misses: cjk 15 (f32 3.0, engine -2.5)
- i8bf16 misses: prose 18 (f32 7.2, engine -1.1); cjk 15 (f32 3.0, engine -0.4)

**Case (a), not (b).** Every int8-form miss sits on a row whose f32 margin is 3-7 bf16 ulps (cjk 15, prose 22, prose
18), no form misses a row above 8 ulps, and W4A8 is wrong-side by at most 2.5 ulps on its one miss at chunk
2048. Its logits cosine against f32 on the missed rows is within 2e-6 of the bf16 twin's (cjk 15), or the twin
misses the same row (prose 18 at chunk 16). That is a few edge rows, not `l0-int8`
systematically off: the near-miss rule decides them, and no fix of the int8 path comes before the int8 speed work.

## Files

`tools/probe/probe_moe_prefill.{cc,cl}`, `tools/probe/moe_w4a8_accuracy.py`, the dump modes in
`tests/golden/ornith_prefill_test.cc` and `tests/runtime/qwen4exp_prefill_test.cc`, `src/kernels/prefill/
pf_moe_sort_mw.cl`. Logs on the box: `~/b70/logs/ornith-moe-prefill/` (probe1, probe_c4096, probe_r2c4096,
probe_r3*, probe_r4, probe_r5, probe_final, probe_fn, numerics_prose), Task 8 in `t9` / `t10`.
