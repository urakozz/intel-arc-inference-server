# Probe: short prefill chunks (open-work 3a) - 2026-11-08

**Question.** Every Qwen3.8 prefill chunk cost ~0.7 s however few rows it had (16 rows 701 ms, 54 rows 709,
256 rows 734, 2048 rows 1161; `b70-decode --bench --prefill-length N`, one card). Where does a short chunk's time
go, and what makes it cheap?

**Timing grade: busy-box timing (one card, `tools/gpu_lock.sh card 1`, the validation run on the other card and
the CPU).** Every comparison is interleaved after a warm-up (`tools/probe/probe_small_chunk.cc`: arms in a
rotating order per round, medians of 5-7 rounds) or run as before / after pairs (`walls.sh` below).

## 1. Where the time goes

`B70_PREFILL_PROFILE=1 b70-decode --bench --prefill-length N` (L0 GPU ms per phase, one chunk):

| phase | 16 rows | 54 | 128 | 256 | 2048 |
|---|---:|---:|---:|---:|---:|
| slab GEMM (`pf_gemm`, the five int4 linears) | 571.9 | 573.1 | 573.7 | 573.3 | 681.1 |
| slab dequant (`pf_dequant_slab`, int4 -> bf16 per 1024 columns) | 96.0 | 96.7 | 97.9 | 98.9 | 124.4 |
| everything else (norms, GDN, attention, head) | ~9 | ~15 | ~24 | ~36 | ~313 |
| total | 676.8 | 684.3 | 695.5 | 708.5 | 1118.5 |

The default prefill backend is `l0` (bf16): per 1024-column slab, a dequant then `pf_gemm`. `pf_gemm` gives every
256 x 256 output tile one work-group (32 sub-groups, each 32 rows x 64 columns walking all of K), so a slab is
4 work-groups for any chunk up to 256 rows, 16 up to 1024 - 4 to 16 of the 32 Xe cores, each doing a whole
256-row tile of DPAS for the rows that exist. Its time is flat from 16 to 1024 rows (575-610 ms for one chunk's
linears, `probe_small_chunk`'s g256 arm) and only 2048 rows (32 work-groups) fill the card.

## 2. The fix: `pf_gemm_small` (bit-identical)

`src/kernels/prefill/pf_gemm_small.cl`: a sub-group owns NA 8-row blocks x NB 16-column atoms, a work-group SN
sub-groups along the columns, and the grid covers only pad(M, 8 NA) rows - many small work-groups over every
core. Each accumulator takes exactly `pf_gemm`'s DPAS chain (per 32-k tile the k16 half 0 then 1, from zero) and a
DPAS output row depends only on its own A row, so the output is **bit-identical** to `pf_gemm` - no rounding
change, no judge needed. Checked: 2,455 probe comparisons (every tile x row count x shape, plain and SiLU, 0 words
differ); `pf_gemm_test` at 1..1025 rows (0 differ); the l0 golden gate's whole printout byte-identical before /
after at chunk 0 and chunk 16 (every cosine to 9 digits; only the load-time line differs).

The sweep (one chunk's five linears with their dequants, ms; the engine's walk - each slab's GEMM right after
its dequant; a GEMM-only arm over one resident slab reads B hot from cache and flatters the small tiles):

| rows | pf_gemm walk | A2_B2_S4 | A4_B2_S4 | A1_B2_S1 | A4_B4_S4 |
|---:|---:|---:|---:|---:|---:|
| 16 | 756 | 221 | 272 | 217 | 374 |
| 64 | 692 | 212 | 256 | 206 | 352 |
| 128 | 681 | 223 | 255 | - | 349 |
| 256 | 688 | 239 | 413 | - | 363 |
| 512 | 712 | 333 | 454 | - | 746 |
| 768 | 725 | 407 | 461 | - | 755 |
| 896 | 750 | 460 | 475 | - | 766 |
| 1024 | 690-765 | 513 | 481 | 2064 | 814 |
| 1280-2048 | 805-823 | 999-1703 | 849-1226 | - | 792-830 |

(column tiles first, the build's order). **Grid order matters:** with the row tiles on grid dimension 0
(`pf_gemm`'s order) the same tiles timed erratically - A4_B2_S4 255 ms at 224 rows, 596 at 256, 316 at 288, 753 at
320 - and column tiles first is smooth and faster everywhere measured. The dequant (~100 ms a chunk, fixed) is
now about half of a short chunk's linear time.

**Built:** `gemm_l0` / `gemm_l0_silu` run an untransposed one-batch GEMM of <= 1024 rows as A2_B2_S4 (to 896 rows)
or A4_B2_S4 (897-1024), `pf_gemm` above (`runtime::prefill::pf_small_tile`). Every model's slab GEMMs go through
`gemm_l0`, so all prefill paths get it; attention's batched GEMMs (L > 1, or transposed B) keep `pf_gemm`.
`-DB70_PF_SMALL_SWEEP=ON` builds the NA 1/2/4 x NB 2/4 x SN 1/2/4 sweep for the probe.

## 3. The engine, before / after

`b70-decode --bench --prefill-length N` (device-side ms, first launch to first id; card 1; main's build and this
branch as back-to-back pairs, two rounds - the rounds agree within 4 ms):

| ids | before | after | |
|---:|---:|---:|---:|
| 1 | 696 | 229 | 3.0x |
| 16 | 700 | 229 | 3.1x |
| 54 | 709 | 238 | 3.0x |
| 128 | 719 | 263 | 2.7x |
| 237 | 729 | 288 | 2.5x |
| 512 | 767 | 407 | 1.9x |
| 896 | 839 | 611 | 1.4x |
| 1024 | 857 | 660 | 1.3x |
| 2048 | 1162 | 1161 | 1.0x |
| 2102 (2048 + 54) | 1874 | 1405 | 1.33x |

`b70-serve` (card 1, `/v1/completions`, max_tokens 1, temperature 0; main's build and this branch alternating,
two rounds each, the same prompts; the server's own `prefix:` prefill_ms, and the client's wall per request):

| request | prefill before | after | wall before | after |
|---|---:|---:|---:|---:|
| cold, 2102 ids | 1885 | 1412 | 2.02 s | 1.54 s |
| follow-up, 4233 ids (the first 2103 cached; 2130 new = 2048 + 82) | 1927 | 1480 | 2.04 s | 1.59 s |
| next turn, 4333 ids (restored at 4096; 237 new) | 741 | 299 | 0.85 s | 0.40 s |

The follow-up's new ids now read at 1,439 t/s (1,105 before).

## 4. What is left

**Update 2026-11-08:** the dequant floor is gone - 3b (`docs/probes/probe-fused-small-chunk-2026-11-08.md`) reads the
int4 weights straight into the DPAS, bitwise identical: one chunk's linears 51-58 ms at 1-16 rows. What remains is
open-work row 3d.

- **The dequant floor.** A short chunk still pays ~100 ms of slab dequant plus ~110 ms of GEMM: 1 row costs
  ~227 ms. A GEMM straight from the int4 weights skips the bf16 slab: `gemm_sm` (spec 28b, decode's small-M
  kernel) takes one chunk's linears at 16 rows in **30-35 ms** (vs ~215 with the dequant). It rounds differently
  (grouped dequant, fp32 group scales, split-K slices), so a <= 16-row chunk through it changes outputs - the f32
  judge's call, and every chunked-prefill bitwise test (chunk 16) would move to the judge. Not built here.
- **Two-card (pp) prefill** goes through the same `gemm_l0`; not timed here.
