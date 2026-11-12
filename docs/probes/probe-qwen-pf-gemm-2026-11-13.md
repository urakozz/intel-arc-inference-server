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

Per shape (probe, section 2: `ctl_gemm` and `ctl_dq` at 2048 rows, one chunk's layers, card 1, medians of 5):

| linear (K x N, layers) | GEMM ms | TFLOP/s | of peak | conversion ms |
|---|---:|---:|---:|---:|
| MLP gate\|\|up, SiLU epilogue (5120 x 34816, 64) | 373.4 | 125.1 | 68.2 % | 42.1 |
| MLP down (17408 x 5120, 64) | 176.5 | 132.4 | 72.2 % | 48.4 |
| delta-net q,k,v,z (5120 x 16384, 48) | 127.7 | 129.2 | 70.4 % | 15.0 |
| delta-net out / attention o_proj (6144 x 5120, 64) | 63.9 | 129.0 | 70.3 % | 7.7 |
| attention q\|\|k\|\|v (5120 x 14336, 16) | 37.2 | 129.2 | 70.4 % | 4.6 |
| **sum** | **778.7** | 128.0 | 69.8 % | **117.8** |

The probe's synthetic weights (uniform random int4 bits, normal activations) run the GEMM ~13 % slower than the real
model (778.7 against the engine's 689.7 ms a chunk, as `probe-prefill-chunk-split-2026-11-10.md` also found); the
conversion agrees (117.8 against 124.4). The ranking and the arm ratios are what the probe is for. So the four mixer
linears are 30 % of the GEMM time and the MLP 70 %; every shape runs within 4 points of the others.

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

**Results, round 1** (card 1, busy box: another session's CPU job, the other card idle; log
`~/b70/logs/qwen-pf-gemm/probe1.log`). Checks: 0 differing values in every slab of `bs_dq` (all five families) and in
the `new` walk's output at 2048 and 4096 rows. Sums over the five families, GPU ms, medians of 5 interleaved rounds:

| arm | 2048 rows | 4096 rows |
|---|---:|---:|
| `ctl` (the production walk) | 898.7 | 1800.0 |
| `new` (`bs_dq` + `r1_gemm`) | 892.7 (-6.0) | 1703.6 (-96.4) |
| `ctl_gemm` | 778.7 | 1657.8 |
| `r1_gemm` | 786.2 (1.010x) | 1602.9 (0.967x) |
| `ctl_dq` | 117.8 | (rows-free) |
| `bs_dq` | 97.2 (0.825x) | |

- **`bs_dq`: FAIL** (-6.0 ms a 2048-row chunk in the walk against the >= 35 ms gate). Alone it is 18 % faster
  (down excepted: its 35 MB slab does not fit the L2, 48.4 -> 48.1), but in the walk the gain is gone: the slab's bytes
  still reach memory before the GEMM reads them. The conversion is bound by bytes, not store messages, at this size
  (~545 GB/s of read + write, as the 2026-09-22 fused-dequant record found).
- **`r1_gemm`: FAIL at 2048 rows** (1.010x, as derived: one wave either way). **At 4096 rows it pays: 0.967x**, and
  with `bs_dq` the 4096-row walk is 1703.6 ms against two 2048-row walks' 1797.4 (-5.2 %, derived from the table).

**Second round (gates written before it):** the tiling knobs `pf_gemm.cl` exposes without a new register budget (the
256 x 256 x 32 tile already holds 224 of 256 registers, so a deeper k-tile or a bigger sub-group tile does not fit):
`p1` / `p3` / `p4` (`PF_DIST`, k-tiles prefetched ahead; production 2) and `nb` (`NO_BARRIER=1`, no per-k-tile
work-group barrier). Each bitwise (same DPAS chain), GEMM alone, 2048 rows. **Gate: <= 0.97x `ctl_gemm` summed over the
five families.**

**Results, round 2** (log `~/b70/logs/qwen-pf-gemm/probe2.log`). Every arm 0 differing values at 2048 rows, all
families. Sums, GPU ms, medians of 5:

| arm | ms | ratio |
|---|---:|---:|
| `ctl_gemm` | 799.1 | 1 |
| `p1` (prefetch 1 ahead) | 798.2 | 0.999 |
| `p3` (3 ahead) | 797.6 | 0.998 |
| `p4` (4 ahead) | 805.7 | 1.008 |
| `nb` (no barrier) | 1069.9 | 1.339 |

**All FAIL** (gate 0.97x). The prefetch distance is flat; the barrier is what keeps the 32 sub-groups' cooperative
prefetch useful (without it, -25 %). Their defines were removed from `pf_gemm.cl` after this round.

## 3. Engine check: is a 4096-row chunk already better? (not a gate)

`--prefill-chunk 4096` against 2048 on main's build, pp4096, `l0`, card 1, interleaved, 3 rounds
(`~/b70/logs/qwen-pf-gemm/chunk4096.out`): 2347.5 / 2347.8 / 2349.2 ms at 2048 against 2349.3 / 2348.9 / 2350.1 at 4096
- **the same**. Converting each weight once per 4096 ids saves ~124 ms, and the 4096-row GEMM loses it again (probe:
1657.8 against 2 x 778.7 = 1557.4 ms, +100). That loss is the A-matrix traffic `r1_gemm` removes.

## 4. Verdict

**Stop (the plan's rule: no arm reaches its gate). Nothing is built.**

- At the engine's 2048-row chunk the bf16 GEMM runs at **78.8 % of the DPAS peak** in the engine and every tiling arm
  the kernel's register budget allows is flat or worse. The conversion (11 % of the prompt pass) is bound by memory
  bytes: storing it differently does not move the walk.
- **The one real gain found is at 4096-row chunks:** `RASTER=1` (bitwise) + `BLOCK_STORE=1` (bitwise) bring the
  4096-row walk 5.2 % under two 2048-row walks in the probe (-96 ms per 4096 ids, synthetic; ~-4 % of the prompt pass,
  derived). Taking it needs the dense model's chunk cap moved to 4096 (`cli/max_len.h` `kPrefillChunkCapRule`, plan
  5d's rule, which then measured +4.2 % at pp4096 and was kept at 2048 against its +5 % bar; today the two caps measure
  the same). That is a session-level choice, outside this task's GEMM / slab scope: recorded as the next step, not
  built. Both defines stay in the production sources (off by default; every production binary byte-identical to
  main's) so the probe and that step can build them.

## 5. Files

- `tools/probe/qwen_pf_gemm_profile.sh` - the engine profile (Task 1).
- `tools/probe/probe_qwen_pf_gemm.cc` - the walk and its arms (`--rows`, `--shape`, `--gemm r1`, `--no-dq`).
- `src/kernels/prefill/pf_gemm.cl` `RASTER`, `src/kernels/prefill/pf_dequant_slab.cl` `BLOCK_STORE` - probe defines,
  default 0.
- Logs: `~/b70/logs/qwen-pf-gemm/` (`profile-l0-*.log`, `plain-l0-*.log`, `probe1.log`, `probe2.log`, `chunk4096.out`).
