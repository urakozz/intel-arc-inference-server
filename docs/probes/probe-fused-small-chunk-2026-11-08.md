# Probe: short prefill chunks reading the int4 weights directly (open-work 3b) - 2026-11-08

**Question.** After 3a (`pf_gemm_small`, [probe-small-chunk-2026-11-08.md](probe-small-chunk-2026-11-08.md)) a
short Qwen3.8 prefill chunk's five int4 linears cost ~210-240 ms, ~100 ms of it `pf_dequant_slab` writing every
weight as a bf16 slab (fixed, whatever the row count). Does a GEMM that reads the int4 weights and converts them in
registers - the slab path's arithmetic, so **bit-identical** - beat the slab walk at short chunks, and up to how
many rows?

The 2026-09-22 probe ([probe-fused-dequant-2026-09-22.md](probe-fused-dequant-2026-09-22.md)) rejected fusion at
2048 rows (the dequant amortises there; fused 100 vs the 123 TFLOP/s break-even) but both its candidates were
bitwise identical to the slab path; its section 7 left "a different tile whose m-subgroup count is smaller" open.
That is this probe.

## 1. The break-even, fixed before measuring

The control is the production walk as built after 3a: per 1024-column slab `pf_dequant_slab` then `gemm_l0`
(`pf_small_tile`: A2_B2_S4 to 896 rows, A4_B2_S4 to 1024). One chunk's five linears (each shape x its layer count:
gate‖up 64, down 64, GDN qkv‖z 48, out_proj / o_proj 64, FA q‖k‖v 16), ms, from 3a's sweep (busy-box grade,
card 1, interleaved medians):

| rows | slab dequant (fixed) | slab walk total (dequant + `pf_gemm_small`) | a fused kernel must beat |
|---:|---:|---:|---:|
| 1 | ~96 | ~220 (16 rows' figure; the engine: 1 row 229 ms a chunk, 16 rows 229) | ~220 |
| 16 | 96 | 221 | 221 |
| 64 | 97 | 212 | 212 |
| 128 | 98 | 223 | 223 |
| 256 | 99 | 239 | 239 |
| 512 | ~105 | 333 | 333 |
| 1024 | ~110 | 481 (A4_B2_S4) | 481 |

**Rule.** The probe times the control (`l0_now`) and every fused candidate interleaved in one process after a
warm-up; a candidate takes a row count only where its median is **at least 5 % below the control's median in that
run** (the 3a numbers above fix the scale, the in-run control is the comparison). Everything else stays on the slab
walk. And only if every fused output equals the control's **bit for bit** (memcmp of every output word, every row
count, every Qwen3.8 shape, plain and SiLU) - otherwise nothing is built.

**What is possible (derived).** One chunk reads 24.3 G int4 weights (12.2 GB, plus 0.76 GB of f16 scales): ~21 ms
at the B70's 608 GB/s, ~26 ms at a realistic 500. The dequant arithmetic is ~7 SIMD16 instructions per 16 weights
(the 09-22 probe's count), ~15 ms over the whole card if each weight is converted once; a register-only kernel
converts it once per row tile, so the ALU term grows with the rows (x M / rows-per-sub-group) while the slab
path's stays flat. `gemm_sm` (decode's small-M kernel, rounds differently) does the 16-row chunk in 30-35 ms, so
~30-40 ms is the floor for <= 16 rows. **Target:** <= 70 ms for one chunk's linears at <= 64 rows (a short turn's
first token well under 100 ms with the ~15-30 ms of norms, GDN, attention and head).

## 2. The kernel: `pf_gemm_q4`

`src/kernels/prefill/pf_gemm_q4.cl` is `pf_gemm_small` (3a) with only the B path replaced, as the 09-22 probe's V1
("registers only") was `pf_gemm` with only the B path replaced: a sub-group owns NA 8-row blocks x NB 16-column
atoms, a work-group SN sub-groups along the columns, the grid is (N / (16 NB SN), pad(M, 8 NA) / 8 NA) - but N is the
linear's **whole** width, one launch a linear instead of a dequant + GEMM per 1024 columns. Per 64-k group a
sub-group loads its columns' 8 int4 words and the group scale (layout 0: one 2D block read of the `[K/8][N]` words
and one block read of the scales; layout 1, the tiled GDN qkv‖z: the tile's 136 words), one group ahead in
registers, and converts them with `pf_dequant_slab`'s statements (the same xor, sign-extension shift,
`(float)q * scale`, single RNE) into the DPAS B fragment the slab's VNNI transform read would give (the 09-22
probe's `pf_dequant8`). Each accumulator takes `pf_gemm_small`'s DPAS chain, so the output is the slab walk's.
NA 8 (64 rows a sub-group, two 32-row A reads) was added for 48-384 rows: the conversion repeats once per row tile,
so the bigger the tile the less it repeats.

**A compiler finding.** The first layout-1 build read k 0..63 of A in every 64-k group (all 16,384 words of the
first check differed; the layout-0 build was bit-identical): the 2D block read of A kept its first x coordinate
across the loop. Moving k into the base pointer (`A + k0`, x = 0; k0 bf16 is a 64-byte multiple) fixed it - the
same arithmetic, so no rounding question. IGC 2.41 on the box; the kernel's comment records it.

## 3. Equality: bitwise, everywhere

`tools/probe/probe_fused_small.cc` compares every candidate tile against the slab walk (`pf_dequant_slab` +
`gemm_l0`, the production launches) as raw 32-bit words over the M real rows before timing anything; a difference
throws. Three runs: **2,276 comparisons** (31 tiles x 1..1024 rows x the five Qwen3.8 shapes, plain for all five,
gate‖up's SiLU epilogue and gate‖up plain too), **0 words differ**. `tests/prefill/pf_gemm_test.cc` now checks
the tiles `pf_q4_tile` binds the same way at 1-384 rows (both layouts, plain and SiLU, K 5120 and 17408): 0 words
differ.

## 4. Times

Busy-box grade (card 1; the other agent's work and CPU jobs on the box). One chunk's five linears (each shape x its
layers), ms, sums of interleaved medians (7 rounds after a warm-up), the slab walk (`l0_now`) and the rule built
(section 5) in the same runs:

| rows | slab walk (dequant + `pf_gemm_small`) | `pf_gemm_q4` rule | gate‖up part | the four plain linears | |
|---:|---:|---:|---:|---:|---:|
| 1 | 241.9 | **51.5** | 15.3 | 36.2 | 4.7x |
| 8 | 240.5 | **50.5** | 15.2 | 35.3 | 4.8x |
| 16 | 236.2 | **58.4** | 25.3 | 33.1 | 4.0x |
| 32 | 236.3 | **59.0** | 25.3 | 33.7 | 4.0x |
| 64 | 237.3 | **76.9** | 39.0 | 38.0 | 3.1x |
| 128 | 249.9 | **113.4** | 64.4 | 49.0 | 2.2x |
| 192 | 254.0 | **154.2** | 90.6 | 63.7 | 1.65x |
| 256 | 270.1 | **213.4** | 111.2 (slab) | 102.2 | 1.27x |
| 384 | 301.1 | **251.2** | 126.2 (slab) | 125.0 | 1.20x |
| 448-768 | 316-399 | (slab walk) | | | 1.0x |

Every row clears the break-even (section 1) by far more than 5 %; the target (<= 70 ms at <= 64 rows) holds to 32
rows (59) and misses at 64 (77). Per linear, where the fused kernel stops winning (ms a linear):

| | gate‖up (SiLU) | down | GDN qkv‖z | out_proj | FA q‖k‖v |
|---|---|---|---|---|---|
| fused / slab at 192 rows | 1.45 / 1.66 | 0.42 / 1.31 | 0.42 / 0.72 | 0.16 / 0.29 | 0.37 / 0.67 |
| at 224 | 1.63 / 1.69 (4 %) | 0.79 / 1.32 | 0.56 / 0.75 | 0.27 / 0.29 | 0.47 / 0.66 |
| at 384 | 2.45 / 1.93 | 0.85 / 1.51 | 0.84 / 0.88 | 0.28 / 0.34 | 0.72 / 0.80 |
| at 448 | | 1.18 / 1.66 | 0.94 / 1.01 | 0.40 / 0.39 | 0.83 / 0.91 |
| at 512 | | 1.45 / 1.54 | 1.12 / 0.94 | 0.43 / 0.39 | 0.86 / 0.89 |

The tiles: NB 1 (16 columns a sub-group) wins the plain shapes - twice the sub-groups of NB 2 over the K walk, and
these shapes (N 5120 for down and out_proj) have few columns; the SiLU epilogue needs NB even. NA 4 x NB 4 and
NA 8 x NB 2 spill (32-46 registers) and lose.

## 5. Built

`runtime::prefill::pf_q4_tile(M, silu)` (`gemm_l0.cc`) and `gemm_q4`; `linear_l0` / `linear_l0_silu` (every Qwen3.8
int4 linear on the default l0 backend) run a chunk of **<= 384 rows** (gate‖up's SiLU form **<= 192**) as one
`pf_gemm_q4` launch, the slab walk above:

| | <= 8 rows | 9-32 | 33-192 | 193-384 | above |
|---|---|---|---|---|---|
| plain (both layouts) | A4_B1_S1 | A4_B1_S1 | A8_B1_S2 | A8_B1_S2 | slab walk |
| gate‖up, SiLU epilogue | A1_B2_S4 | A4_B2_S1 | A4_B2_S1 | slab walk | slab walk |

Automatic, no flag or variable; the build carries those six binaries (`-DB70_PF_Q4_SWEEP=ON` the probe's sweep).
The l0-int8 backend (`linear_i8`), the MTP head's bf16 linears and other models' own walks are unchanged.

## 6. The engine, before / after

`b70-decode --bench --prefill-length N --tg 1` (device-side ms, first launch to first id; card 1; main's build
`122fb36d` and this branch as back-to-back pairs, two rounds in alternating order - the rounds agree within 3 ms):

| ids | before | after | |
|---:|---:|---:|---:|
| 1 | 228 | **73** | 3.1x |
| 54 | 237 | **109** | 2.2x |
| 237 (gate‖up on the slab) | 287 | **259** | 1.11x |
| 512 | 408 | 408 | - |
| 1024 | 656 | 656 | - |
| 2048 | 1164 | 1163 | - |
| 2102 (2048 + 54) | 1405 | **1266** | 1.11x |

A short turn's first token: 73 ms at 1 id, 109 at 54 (the target "well under 100 ms" holds for the shortest
turns only; at 54 ids the linears are ~77 ms of the 109). 2102 ids: 1266 ms, below the 1.4 s target.

**Tests (card 1, this branch).** `pf_gemm_test` (the new bitwise q4 cases, 0 words differ everywhere);
prefill_smoke (+ cap4096; the launch arithmetic re-derived: a chunk of <= 192 rows is 1089 launches instead of
8449, and the live counter agrees), prefill_gate_l0 (+ c16, i8head, kv8), prefill_consistency_l0 (+ kv8),
prefill_determinism_l0 (+ kv8), prefill_split_l0, prefix_gpu_split_l0, snapshot, mtp_prefill, Ornith's
ornith_prefill_l0 / l0-int8 and prefill_split_ornith_l0 (its mixer linears take the same path): all pass. The
four l0 golden gates' printouts are **byte-identical** to main's build (every line; only the load-time lines
differ).

`b70-serve` (card 1, `--spec off`, `/v1/completions`, max_tokens 1, temperature 0; main's build and this branch
alternating, two rounds each, the same prompts; the server's own `prefix:` prefill_ms and the client's wall):

| request | prefill before | after | wall before | after |
|---|---:|---:|---:|---:|
| cold, 2058 ids (2048 + 10) | 1392 | **1241** | 1.49 s | 1.35 s |
| follow-up, 4150 ids (2091 new = 2048 + 43) | 1437 | **1308** | 1.53 s | 1.40 s |
| next turn, 4359 ids (208 new; gate‖up on the slab) | 292 | **267** | 0.36 s | 0.34 s |

## 7. What is left

- **193-384 rows and gate‖up.** gate‖up (47 % of the weights) goes back to the slab walk above 192 rows: its SiLU
  epilogue needs two atoms a sub-group, and NA 8 x NB 2 spills. A 237-id turn gains only 11 %. A tile that keeps
  64 rows x 32 columns without spilling (or the gate / up halves on two sub-groups with the epilogue through SLM)
  would carry gate‖up to ~384 rows.
- **A 54-id turn is 109 ms**, not "well under 100": the linears are ~77 ms of it at 33-64 rows (the conversion
  repeats per 64-row tile and the DPAS work grows); the rest (norms, GDN, attention, head) ~30 ms.
- `gemm_sm` (30-35 ms at 16 rows, rounds differently) is not needed for <= 16 rows any more: `pf_gemm_q4` does
  one chunk's linears in ~51-58 ms there, bit-identical.
- Two-card (pp) prefill runs the same `linear_l0` and gets the same kernel; not timed here.
