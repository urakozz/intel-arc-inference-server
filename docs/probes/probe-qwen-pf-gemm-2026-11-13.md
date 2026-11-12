# Probe: Qwen3.8's bf16 prefill linears - where the time goes and what is left (2026-11-13)

**Question.** Qwen3.8-27B's prompt pass on `l0` (bf16, one card) spends most of its time in the int4 linears, which
run as a slab walk: per 1024-column slab, `pf_dequant_slab` converts the int4 weight to a bf16 slab, then `pf_gemm`
(the DPAS GEMM, DPAS = the matrix-engine instruction) multiplies the chunk's rows by it. Is any of that worth
taking - in the GEMM itself, or in the conversion around it? Branch `qwen-pf-gemm`.

Every number is **measured** unless marked **derived**.

## 1. The profile (Task 1)

One B70 (card 1, `tools/gpu_lock.sh card 1`), main's build (cf22d86a), real weights
(`urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`), `b70-decode --bench --prefill-backend l0 --max-len 32768 --tg 16`,
bf16 KV. Busy box (another session's CPU job: load average ~58; the other card idle). `B70_PREFILL_PROFILE=2`: the
phase table plus the per-launch trace (L0 kernel timestamps; one wait per slab, so the instrumented wall is ~7 % above
the plain run). Script `tools/probe/qwen_pf_gemm_profile.sh`, logs `~/b70/logs/qwen-pf-gemm/profile-l0-*.log`.

**The engine runs 2048-row chunks here** (`chunk 2048` on the pp line; the 4096-row cap is not chosen at
`--max-len 32768`), so pp4096 is two chunks and pp16384 eight.

| prompt | plain wall | profiled wall | kernel sum |
|---|---:|---:|---:|
| 4096 | **2345.4 ms, 1,746 t/s** | 2520.6 ms | 2265.7 ms |
| 16384 | **10444.9 ms, 1,569 t/s** | 11140.9 ms | 10071.6 ms |

| group (L0 kernel ms) | pp4096 | share | pp16384 | share |
|---|---:|---:|---:|---:|
| **bf16 GEMM** (`pf_gemm`, 7616 / 30464 launches) | **1379.4** | **60.9 %** | **5491.5** | **54.5 %** |
| **int4 -> bf16 slab** (`pf_dequant_slab`, same counts) | **248.7** | **11.0 %** | **994.0** | **9.9 %** |
| delta net (`ab_proj` + ten `gdn_*` kernels, 48 layers) | 427.4 | 18.9 % | 1712.5 | 17.0 % |
| attention (`attn_prep` + `attn_flash` + `attn_gate`, 16 layers) | 104.7 | 4.6 % | 1458.6 | 14.5 % |
| norms (`pf_res_fold` + `pf_norm_finish`) | 103.0 | 4.5 % | 411.9 | 4.1 % |
| head | 2.2 | 0.1 % | 2.2 | 0.0 % |

**The GEMM's rate (derived):** one prompt row through the int4 linears is 48.65 GFLOP
(`probe-prefill-chunk-split-2026-11-10.md`): 199.3 TFLOP at pp4096 in 1379.4 ms = **144.5 TFLOP/s, 78.8 % of the
183.45 TFLOP/s bf16 DPAS peak**; pp16384 797.1 TFLOP in 5491.5 ms = 145.2 TFLOP/s, 79.1 %. The conversion is
124 ms a 2048-row chunk whatever the rows (it converts every weight once a chunk).

The engine's trace names kernels, not shapes (`pf_gemm` and `pf_gemm` with the SiLU epilogue share the entry name),
so the per-shape split is the probe's (section 2, synthetic weights at the same shapes, 2048 rows).

PER_SHAPE_TABLE

## 2. The probe arms (Task 2)

`tools/probe/probe_qwen_pf_gemm.cc`: probe_prefill_split's distinct-layer walk (each of the five linear families with
one live allocation of independent synthetic weights per layer, the model's shapes and layer counts), the production
walk against arms built from the same two sources with one define each. Both defines default to off, and at off
every production binary is byte-identical to main's (checked: `cmp` of all 5368 kernel binaries).

- **`bs_dq`** (`pf_dequant_slab.cl` `BLOCK_STORE=1`): the conversion stores each 32-bit word's eight values as one 2D
  block message (8 rows x 16 columns, lane = column) instead of eight 32-byte row stores - 8x fewer store messages,
  the same values. Why: the slab is 10.5 MB of bf16 a 5120-K slab written by 64 scalar row stores per work-item; the
  repo's own finding (`pf_gemm.cl`'s epilogue note) is that this card's stores are bound by the message count.
- **`r1_gemm`** (`pf_gemm.cl` `RASTER=1`): the work-group order N-fastest, so a wave of 32 work-groups covers 8 row
  blocks x 4 column blocks of a slab instead of 16 x 2 and each A row block is read once a slab. The same tiles, the
  same DPAS chain. This is the "strip-ordered numbering" idea phase 3 dropped for Ornith (its A, 2048 x 2048 bf16,
  fits the 24 MB L2); Qwen3.8's A is 20-71 MB a 2048-row chunk. **But at 2048 rows a slab is 8 x 4 = 32 work-groups,
  one wave, and every A row block is read once in either order** (derived), so it can only matter at 4096-row chunks;
  measured at both for the record.
- **`new`**: `bs_dq` + `r1_gemm` as a walk.

Checks before any time: layer 0, every slab of `bs_dq` equal to `ctl_dq`'s byte for byte, and the `new` walk's output
equal to the control walk's byte for byte (all rows, all columns).

**Gates (written before the run):**
- `bs_dq`: bitwise, and the walk with it (`new` at 2048 rows, where `r1_gemm` is the control's order in effect) saves
  **>= 35 ms a 2048-row chunk** summed over the five families - 3 % of a chunk's ~1170 ms (derived from pp4096's
  2345 ms).
- `r1_gemm`: bitwise, and **<= 0.97x** `ctl_gemm` summed at the engine's chunk (2048 rows); else recorded only.
- Task 3 builds only what passes, and only if the sum is >= 3 % of the whole prefill.

PROBE_RESULTS
