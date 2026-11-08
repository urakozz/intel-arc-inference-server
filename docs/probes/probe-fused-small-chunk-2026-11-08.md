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
gate||up 64, down 64, GDN qkv||z 48, out_proj / o_proj 64, FA q||k||v 16), ms, from 3a's sweep (busy-box grade,
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
