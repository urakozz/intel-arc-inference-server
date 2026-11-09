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

## 8. 3d: short-prefill leftovers — 2026-11-09

The acceptance rule was committed before timing (`338c74dc`): select a row
count only when the candidate is at least 5% below the slab walk in the same
interleaved run, with every output word bitwise identical. These runs hold
the whole-box lock, use nine rotating rounds after warm-up, and time the
actual per-slab dequant/GEMM walk rather than a resident-slab GEMM.

### Candidate selection

The first candidate's 256-GRF option was already present for NA × NB >= 4.
Its A8_B2 S1/S2 builds lose at every sampled row count. A4_B2 with S2/S4
also misses the rule: its best observed gain is 4.42% at 224 rows.
Each family has 40 plain/SiLU bitwise comparisons with zero differing words.

The third candidate separates gate and up into NB1 launches. The original
paired epilogue rounds both fp32 accumulators to bf16 before SiLU, so the
first launch can store gate bf16 in the existing output X; the second reads
that exact value, rounds up, and executes the same SiLU-RNE, multiplication
and final-RNE chain. DPAS accumulation and q4 conversion orders are unchanged.
The existing MLP input is `mixer_out`, distinct from output `x`, so storing
the gate cannot overwrite the up launch's input. No extra scratch is needed.

An initial split S1/S2 sweep found only a marginal 256-row winner. S4 was then
checked at the boundary counts and confirmed at **every row from 193 through
256**, with six-decimal timing output to avoid rounding the 5% decision.
All 64 counts pass bitwise and clear the rule: gains range from **5.0825% at
217 to 8.5483% at 251**. Different runs' controls moved; only same-run ratios
were used for selection.

Representative final-confirmation medians, ms per gate/up linear (64 per chunk):

| rows | slab walk | split S4 | gain |
|---:|---:|---:|---:|
| 193 | 1.745625 | 1.640521 | 6.0210% |
| 217 | 1.755417 | 1.666198 | 5.0825% |
| 237 | 1.798177 | 1.668125 | 7.2324% |
| 250 | 1.789219 | 1.643073 | 8.1681% |
| 256 | 1.784687 | 1.651406 | 7.4680% |

The upper-range control run rejects expansion: at 257 / 272 / 288 / 300 /
320 / 352 / 384, S4 is respectively 11.12 / 9.78 / 10.87 / 8.74 / 10.47 /
28.33 / 26.78% slower than its in-run slab control. These counts retain the
slab path; no additional work item is created.

### Two-pass integration: rejected by the engine check

The intermediate build `8a5ccb1c` selects A8_B1_S4 split gate/up at 193–256
rows. The default build adds only `_GATE` and `_UP_SILU` for this tile; S1/S2
remain sweep controls. The four plain linears and all other SiLU ranges keep
their previous selection. There is no engine flag, environment variable or
additional buffer.

`pf_gemm_test` first reproduced the missing 193-row dispatch (RED at the
null tile), then passed the new real slab-walk comparisons and its full
existing suite. Bounds at 257 / 300 / 384 require the slab path. Shared launch
accounting charges two gate/up launches: a 250-row Flash chunk is 1153 launches
(1158 including the head), while 300 stays 5377. The smoke test checks the
actual 250-row engine counter against this prediction.

The default build with the sweep disabled passes all 17 named prefill tests.
All four golden forms (base, chunk 16, int8 head, int8 KV) match the baseline
byte for byte after masking only numeric loader/quantisation elapsed times.
However, actual engine timings reject this dispatch:

| request | baseline prefill ms | two-pass prefill ms |
|---|---:|---:|
| CLI 237 ids | 254.0 | 256.2 |
| CLI 300 ids | 288.7 | 290.4 |
| CLI 384 ids | 314.0 | 314.7 |
| warm server, cached 200 new ids | 246.9 | 250.9 |
| warm server, fresh 250 ids | 247.0 | 250.2 |

These are medians from three alternating rounds. The server uses int8 KV,
one card and one generated id; matching prompt and generated ids are checked.
The warm requests are also slower, so cold module loading alone does not
explain the result. The primitive probe reuses one 94.7 MB gate/up allocation
and scales its time by 64; the engine walks 64 distinct matrices (6.06 GB)
and queues surrounding work continuously. Its ratios cannot establish an
engine gain. The two-pass tile also adds roughly 1.14 GB of logical X
reads/writes per 250-row chunk. No hardware traffic attribution is claimed.

This dispatch is not accepted for the final change. Before recording no-go,
test the one-launch paired-subgroup/SLM alternative already suggested in
section 7, under the same bitwise and 5% rule and the actual engine check.
No additional work item is created. Raw logs are under `~/b70/logs/3d`: `large-grf.log`, `smaller.log`,
`split.log`, `split-dense.log`, `split-confirm.log`, `split-upper.log`,
`selector-{red,green}.log`, `final-gates/`, `pp-final/` and `server-pairs/`.

### One-launch paired-subgroup SLM candidate

A8_B1_S4 gives even subgroups the gate atom and adjacent odd subgroups its
up atom. Each pair exchanges 64 × 16 rounded gate bf16 values through SLM
(4 KiB per workgroup), then the odd subgroup runs the unchanged
up-RNE / SiLU-RNE / multiply / final-RNE epilogue. Every subgroup reaches
the workgroup barrier. Physical adjacent gate/up weight columns and the
DPAS accumulation order remain unchanged; there is one launch and no global
gate intermediate.

The initial S2/S4 comparison passes all 14 bitwise checks. S4 is then
confirmed at **every row from 193 through 384**, with 192/192 bitwise
comparisons and zero differing words. All **128 rows from 193 through 320**
clear the fixed 5% bar: 16.0293% at 263 through 26.8091% at 247.
The selector uses that contiguous range. Row 321 immediately misses;
isolated upper cells barely above 5% do not extend the range.

| rows | slab walk ms/linear | S4 SLM ms/linear | reduction |
|---:|---:|---:|---:|
| 193 | 1.835052 | 1.370000 | 25.3427% |
| 237 | 1.789583 | 1.340990 | 25.0669% |
| 250 | 1.804323 | 1.339635 | 25.7541% |
| 263 | 1.953125 | 1.640052 | 16.0293% |
| 300 | 2.000573 | 1.640156 | 18.0157% |
| 320 | 2.023281 | 1.648125 | 18.5420% |
| 321 | 2.022604 | 1.942813 | 3.9450% |
| 384 | 2.094792 | 1.995781 | 4.7265% |

The real selector regression first fails against the old host library at
300 rows (`t != nullptr`), then the full GEMM suite passes after integration
in signed `2fb9d99e`. The default build adds only the S4 SLM module. S2 SLM
and all two-pass modules remain sweep controls. A selected linear has one
launch again. The smoke test pins the 320/321 boundary and checks actual
250/300-row engine counters: 1089 per Flash chunk, 1094 including the head.
The composed forms add 176 launches at 250 rows and 240 at 300.

### Accepted engine and server confirmation

The same immutable baseline and the final default build alternate over three
rounds after warm-up, under the whole-box lock. CLI times are host wall
around prefill, with int8 output head, bf16 KV, chunk width 2048 and one
generated id. The printed `l0 pp` row denotes prompt processing even when
the command selects one card. Medians and median paired time reductions:

| CLI prompt ids | baseline ms | SLM ms | paired reduction |
|---:|---:|---:|---:|
| 237 | 254.9 | 232.2 | 8.93% |
| 300 | 289.4 | 269.5 | 6.88% |
| 384 | 313.6 | 313.8 | -0.06% |

The server comparison uses one card, int8 KV, `--spec off`, temperature 0
and one generated id. Each process first warms separate 2048/2248/250-id
prompts. The measured 200-id follow-up restores exactly 2048 prompt ids;
the fresh 250-id request restarts at zero. All 18 records are complete,
with identical prompt and generated ids across builds in each round.

| server request | baseline prefill ms | SLM prefill ms | paired reduction | baseline client ms | SLM client ms |
|---|---:|---:|---:|---:|---:|
| cold 2048 ids | 1190.1 | 1193.9 | -0.34% | 1264.8 | 1267.4 |
| cached, 200 new ids | 247.1 | 226.4 | 8.38% | 324.2 | 304.0 |
| fresh 250 ids | 246.5 | 227.3 | 7.83% | 311.7 | 293.4 |

The SLM candidate confirms its gain in both CLI and warm serving, unlike
the rejected two-pass route. The unchanged 384-row slab and 2048-row paths
serve as controls. These probe comparisons are diagnostic measurements,
not a new headline record. Raw logs under `~/b70/logs/3d`: `slm-first.log`,
`slm-confirm.log`, `slm-selector-{red,green}.log`, `slm-engine/walls/`,
`slm-engine/server/` and the validated `slm-engine/summary.json`.
