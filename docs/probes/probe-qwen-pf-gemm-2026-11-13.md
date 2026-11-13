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

## 6. Built: the 4096-row walk in the engine (open-work 18, branch `qwen-chunk-4096`)

**What is built.** Two variants beside the plain binaries, each one define (bit-identical by construction: the same
tiles, DPAS chain and stored values): `pf_gemm_T0_R` / `pf_gemm_T0_SILU_R` (`RASTER=1`) and
`pf_dequant_slab_K*_N*_L*_BS` (`BLOCK_STORE=1`) at the five one-card shapes and the five tensor-parallel card shapes.
`runtime::prefill::linear_l0` / `linear_l0_silu` bind them for a dense model (Qwen3.8) when the chunk is wider than
2048 rows; a chunk up to 2048 rows, the MoE models' dense linears and every other model keep the plain binaries.

### Task 1: the engine measurement (before any default moved)

One B70 (card 1, `tools/gpu_lock.sh card 1`), `b70-decode --bench --prefill-backend l0 --max-len 34816 --tg 16`,
bf16 KV. **Busy box**: another session's tests on card 0 throughout. Main's build (e7e3c5bb, its own tree) at
`--prefill-chunk 2048` against the branch's at `--prefill-chunk 4096`, interleaved pairs after one warm-up pair, three
rounds (`tools/probe/qwen_chunk4096_bench.sh`, logs `~/b70/logs/qwen-chunk-4096/r*-l0-010239-*.log`). ms, medians of 3:

| prompt | main, 2048-row chunks | branch, 4096-row chunks | change | rounds (main / branch) |
|---|---:|---:|---:|---|
| 4096 | 2369.0 (1,729 t/s) | **2235.2 (1,832 t/s)** | **-5.6 %** | 2360.3 2369.0 2369.2 / 2227.8 2238.3 2235.2 |
| 16384 | 10466.8 (1,565 t/s) | **9882.5 (1,658 t/s)** | **-5.6 %** | 10452.1 10466.8 10472.3 / 9878.0 9882.5 9888.5 |
| 32768 | 23522.0 (1,393 t/s) | **22190.3 (1,477 t/s)** | **-5.7 %** | 23524.3 23515.6 23522.0 / 22190.3 22187.7 22197.3 |

**Gate (>= 3 % faster at pp4096, no slower at pp16384, repeatable): PASS** - every pair faster, by 5.5-5.7 %.

**Control** (where the gain comes from): main's build at `--prefill-chunk 4096` against the branch's, one round
(`bench-ctl.out`): pp4096 2354.8 / 2357.1 against 2231.9 / 2230.3 ms (main's 4096-row chunk is its 2048-row time, as
section 3 found); pp16384 10357.7 against 9876.3. So the 4096-row chunk alone is ~0.5 % at pp4096 (~1 % at pp16384,
half the attention launches), and the two variants are the rest.

**The phase table** (`B70_PREFILL_PROFILE=1`, one run each; L0 kernel ms; the instrumented walls 2537.9 / 2316.2 and
11012.9 / 10160.6 ms):

| phase | pp4096 main (2048) | pp4096 branch (4096) | pp16384 main | pp16384 branch |
|---|---:|---:|---:|---:|
| `slab_gemm` (`pf_gemm`) | 1382.6 | 1418.2 | 5534.9 | 5661.5 |
| `slab_dequant` (int4 -> bf16 slab) | 249.2 | **114.9** | 999.5 | **459.7** |
| norms | 103.3 | 115.6 | 415.0 | 463.0 |
| delta net (`ab_proj` + `gdn_*`) | 427.8 | 432.4 | 1727.1 | 1732.1 |
| attention (`attn_prep` + `attn_flash` + `attn_gate`) | 104.5 | 102.1 | 1468.5 | 1362.7 |
| **total** | 2269.7 | **2185.4** | 10147.5 | **9681.4** |
| launches | 16903 | 8454 | 67597 | 33801 |

The conversion is paid once per 4096 ids instead of twice (-134 ms at pp4096) and its block stores take it below
half; the 4096-row GEMM with the N-fastest order is 2.6 % over two 2048-row GEMMs (without the order +6 % in the
probe, section 3); flash attention gains from half the chunks at depth.

**Bitwise:** `prefill_split_l0_cap4096_test` PASS on the branch (the engine's own rule: a chunk boundary at a multiple
of 64 is bitwise; the 4096-row chunks run the new binaries).

**`l0-int8`** (the int8 prompt pass, which follows the same session cap; its int8 GEMMs do not use these variants),
same protocol (`r*-l0-int8-011501-*.log`), medians of 3: pp4096 1817.4 -> **1753.8 ms (-3.5 %)**, pp16384 8273.8 ->
**7934.6 (-4.1 %)**, pp32768 19214.9 -> **18341.3 (-4.5 %)**; every pair faster (the h8 quantiser and the slab of the
head's linears amortise over twice the rows).

### Task 2: the default, and what it costs

**Context and slots: nothing.** The rule is the MoE models' (max_len and every slot planned at 2048, 4096 taken only
when the plan at those lengths has room; with `--max-num-seqs` the slots' plan, not slot 0's alone). b70-serve's
start-up on one card, main against the branch with the rule on (`tools/probe/qwen_chunk4096_plans.sh`,
`~/b70/logs/qwen-chunk-4096/plan-*.log`):

| recipe | context / slots (main = branch) | chunk with the rule |
|---|---|---:|
| default (bf16 cache) | 171,008 | 2048 (no room) |
| `--kv-cache int8` | 262,144 | **4096** (0.70 GB more scratch; 2.48 GB left, the 1.5 GB reserve inside it) |
| `--prefill-backend l0-int8` | 169,984 | 2048 |
| `--max-num-seqs 4` | 46,080 + 3 x 32,768 | 2048 |
| `--max-num-seqs 4 --kv-cache int8` | 167,936 + 3 x 32,768 | 2048 |

So only the full-context int8-cache recipe (and replica cards in that form, and any shorter explicit `--max-len`)
would take 4096.

**The served short turn deep in a session: slower - the stop.** llama-benchy through b70-serve `--kv-cache int8`
(one card, the server's own chunk cap), a 64-id turn after a 32,768-id context with the prefix cache
(`--depth 32768 --pp 64 --enable-prefix-caching`, 3 runs a server, interleaved main / branch; `tools/probe/qwen_chunk4096_turn.sh`,
`turn-*.log`, `turn-serve-*.log`), time to the first token, ms:

| server | context pass (cold, 32,770 ids) | the 64-id turn | the turn's restart point |
|---|---:|---:|---|
| main (2048) | 24,455 | 2,093 | 30,720 (2,112 rows prefilled) |
| branch (4096) | **22,983 (-6.0 %)** | **3,682 (1.76x)** | 28,672 (4,160 rows) |
| main, `--prefix-split-last` (2 rounds) | 24,502 | 2,161 / 2,207 | 30,720 |
| branch, `--prefix-split-last` (2 rounds) | **23,075 (-5.8 %)** | **3,734 / 3,706 (1.7x)** | 28,672 |

The cache's state snapshots sit at chunk ends, so with 4096-row chunks they are 4096 positions apart (plan 5d's
ruling for a chunk wider than a block). A hit that restores from a block snapshot - here every time: the client's
second request is not an extension of the first prompt's end, so even `--prefix-split-last`'s prompt-end snapshot
does not apply - restarts up to 4,095 rows back instead of 2,047, and those rows cost more than the 4096-row chunk
saves. In an agentic session (long context, a prefix hit every turn) that is the common case. **The default stays
2048 for Qwen3.8**; the variants stay built and bound for any 4096-row chunk (`--prefill-chunk 4096`, the cap4096
tests), bit-identical. The slot-aware fit check stays (it is what a MoE model beside `--max-num-seqs` now uses).

**Open question (for the operator).** (a) Keep 2048 (as built). (b) 4096-row chunks for the bulk of a long prompt
and 2048-row chunks with a snapshot at every block end over its last ~6k positions - the restart points near a
prompt's end as today, the cold pass most of the -5.6 % (derived; a change to `runtime/prefill_chunks.h`'s stride
rule, which Ornith and the two-card pipeline share). (c) Take 4096 with the coarser restarts (the MoE models' choice).

### Task 3: the gates (box, card 1, `~/b70/logs/qwen-chunk-4096/gates.log`, `final.log`)

All PASS on the branch: `pf_gemm_test`, `linear_l0_lazy_test`, `pf_dequant_slab_test`, `golden_gate_test`,
`golden_gate_i8head_test`, `prefill_split_l0_test`, `prefill_split_l0_cap4096_test`,
`prefill_split_l0-int8_cap4096_test`, `snapshot_test`, `prefill_gate_l0_test`, `prefill_gate_l0_i8head_test`,
`prefill_gate_l0_c16_test`, `prefill_gate_l0_cap4096_test`, `prefill_gate_int8_cap4096_test`,
`prefill_consistency_l0_cap4096_test`, `prefill_determinism_l0_cap4096_test`, `prefill_smoke_test`,
`prefill_smoke_cap4096_test`, `prefill_gate_long_test`, `golden_server_test`, `prefix_gpu_l0_test`,
`prefix_gpu_int8_test`, `mtp_prefill_cap4096_test`, `mtp_verify_test`, `slots_test`, `memory_plan_box_test`, and
the new `prefix_gpu_l0_cap4096_test` (the prefix cache at 4096-row chunks: every sequence's 32 ids exact or within
the one near-tie allowance, as at 2048; sequence f now expects a cold start - no block snapshot below 4096).

**Kernel binaries against main's build** (`cmp`, the two trees): 2,437 identical, 12 new (the variants), and 254
Bolt binaries that differ in both directions from every other tree too - they carry the tree's include path
(`-I .../src/kernels/bolt`) in their build options, and their compiled code shifts with it (rebuilt in place they
are byte-stable). Nothing of Bolt's sources or command lines changed.

Not run: `--parallel tp` / `pp` prefill at 4096 (they stay 2048 by default; an explicit `--prefill-chunk 4096` under
`tp` binds the tensor-parallel shapes' block-store conversions, bit-identical by construction, untested on the box).
