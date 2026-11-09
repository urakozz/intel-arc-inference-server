# Probe: GDN prefill per kernel on Qwen3.8 (open-work 3m) - 2026-11-10

**Question.** 3k ([record](probe-prefill-chunk-split-2026-11-10.md)) found the 48 GDN layers' chunk recurrence
(GDN: Gated DeltaNet, the model's linear-attention mixer) to be the second share of a prefill chunk on the default
`l0` backend: 237 ms of a 1,190 ms 2048-row chunk, 1.8 s of a cold 15,257-id prompt. Which of its eleven kernels
(the `a||b` projection and `gdn_chunk`'s ten launches) is worth a build item?

**Rule, fixed in the row and written here before measuring.** GO for a build item only for a kernel that is
**>= 5 % of the chunk's time** and **below ~60 % of its bound**. The bound is bandwidth or compute, whichever binds
for that kernel: its bytes and FLOPs per launch are computed from the shapes, the time each would take at the
card's measured peak (590 GB/s; 183.45 TFLOP/s bf16 DPAS for a DPAS kernel, the measured fp32 vector rate for a
vector kernel) is computed, and the larger of the two is the bound. Each GO carries an estimated saving. An exact
(bitwise) change comes first; a reassociation only with the f32 judge (the engine judged against an f32 reference,
memory ruling 2026-10-23).

**GDN's numerics are not on the table.** The operator's ruling of 2026-10-31 is that `pf_gdn_wu` keeps its four
intermediates Q1-Q4 in fp32 (accuracy over FLA parity); that change is parked on branch `g0-fix-gdn-fp32` (main
still stores `w` / `u` bf16). Nothing proposed here may add a rounding, and anything proposed must not stand in
the way of the parked fp32 `w` / `u`.

**Verdict: one GO, narrowly - the scan (`pf_gdn_scan_dpas_split`), for its residency.** It is the only GDN kernel
over 5 % of the chunk (63.7 ms of 1,189.9 = **5.4 %**; 4.7 % of the 15,257-id prompt) and runs at **19 % of its
bound** (bandwidth: 149 MB a launch, 253 us at 590 GB/s, against 1,327 us). The probe shows why: its 192 work-groups
do not fit on the card at once (40,448 B of SLM each: 3 per Xe-core, 96 resident), so the launch runs in two waves
of one work-group's whole sequential walk. Estimated saving of an exact, all-resident remap: **~25 ms of a 2048-row
chunk (~2 %), ~190 ms of the 15k prompt**. Every other kernel is a no-go: under 5 % each (the next are `pf_gdn_wu`
3.5 %, `pf_gdn_solve` 3.1 %, `pf_ab_proj` 2.0 %), and the two large-byte kernels (conv, gated head) already run at
62-64 % of bandwidth. The proposed item is open-work 3n (Now, in 3m's place), plan only.

Every number is **measured** unless marked **derived**.

## 1. Method

**Locks.** Building is CPU only (`tools/box.sh build`, no GPU). The smoke checks (both new probes: the fma checksum,
the scan's bitwise sanity) ran under `tools/gpu_lock.sh card <N> -- ...` (one card; other agents may hold the other).
The final timed numbers ran under `tools/gpu_lock.sh box -- tools/probe/gdn_prefill_3m.sh`: both cards idle,
nothing else on the GPU, the arms interleaved after a warm-up. `ZE_AFFINITY_MASK` was never set by hand.

**The engine, real weights.** `b70-decode --bench --prefill-length 2048 | 15257 --prefill-backend l0
--kv-cache bf16 --max-len 32768 --tg 16 --parallel 1` on `urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ`.
`B70_PREFILL_PROFILE=1` gives every kernel of the walk its own L0 GPU timestamp bucket: `ab_proj` and the ten
`gdn_*` launches are separate rows, summed over the 48 GDN layers, so the per-kernel split is the engine's own,
over all 48 layers' real tensors (distinct layers by construction). One warm-up, then three rounds of four arms
(2048 and 15,257 ids, each plain and profiled), the arm order rotating each round. The plain runs give the walls;
the profiled runs give the per-kernel times (median of three).

**The peaks.** `probe_dpas_rate --sweep` (183.45 TFLOP/s bf16 since 2026-09-22) and a new
`tools/probe/probe_fma_rate` (the sustained fp32 vector `fma` rate: 16 independent register chains per work-item,
probe_dpas_rate's calibration and replay protocol), before the run and the fma rate again after it.

**Bytes and FLOPs (derived)** per launch from the shapes in the kernels (`src/kernels/prefill/pf_gdn_conv.cl`,
`pf_gdn_wy.cl`, `pf_gdn_scan.cl`, `pf_gated_head.cl`, `pf_gemv_bf16.cl`) and the scratch layout
(`src/runtime/buffer_sizes.cc`). Bytes are the minimum DRAM traffic: every distinct tensor read once and written
once (re-reads that hit the 24 MB L2 are not counted), so an achieved GB/s is a lower bound on the kernel's real
traffic rate. FLOPs count each `fma` as 2.

**The run.** Whole-box lock 23:34:34-23:39:00 (box clock, 2026-11-09), 4 min 26 s; load average 2.9-3.5 (a
desktop indexer); no CPU reference job. Before it, a card-lock smoke run on card 1 (~1 min: both probes, the scan
sanity PASS). Logs: `~/b70/logs/3m/` (`run.log`, `fma-pre.log` / `fma-post.log`, `dpas-pre.log`, `scan-occ.log`,
`plain-*`, `profile-*`, `smoke-card1.log`).

## 2. The peaks

| | before | after |
|---|---:|---:|
| fp32 vector `fma` (`probe_fma_rate`, best grid) | **22.84 TFLOP/s** (99.6 % of the clock-derived 22.94) | 22.84 |
| bf16 DPAS (`probe_dpas_rate`) | **183.43 TFLOP/s** | - |
| DRAM read bandwidth (docs/01-hardware.md) | 590 GB/s | - |

## 3. The walls (unprofiled, `l0`, bf16 KV)

| prompt | rounds | median |
|---|---|---:|
| 2048 ids (one chunk) | 1192.6 / 1189.8 / 1189.9 ms | **1189.9 ms** (1,721 t/s) |
| 15,257 ids (7 chunks + 921 rows) | 10144.8 / 10135.2 / 10123.9 ms | **10135.2 ms** (1,505 t/s) |

3k's 1189.7 / 10120.1 reproduce within 0.2 %. Profiled walls are 1262-1267 / 10640-10658 ms (the per-kernel waits);
the kernel times below are L0 timestamps, which the waits do not inflate.

## 4. Per kernel, 2048-row chunk

Times: the 48 layers' L0 GPU time, median of three profiled runs (the three agree within 1.3 % on every row).
Bytes and FLOPs are **derived** per launch (one layer, 2048 rows); "bound" is the larger of bytes / 590 GB/s and
FLOPs / the peak of the engine the kernel uses (fp32 vector 22.84 or DPAS 183.43); % of bound = bound / measured.

| kernel (launch) | ms / chunk | share of 1189.9 | us / launch | MB / launch | GFLOP / launch | GB/s | TFLOP/s | binds | bound us | % of bound |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `pf_ab_proj` (a\|\|b GEMV) | 23.3 | 2.0 % | 485 | 23.3 | 2.684 | 48 | 5.53 | fp32 | 117.5 | 24 % |
| `pf_gdn_seed` | 0.1 | 0.0 % | 2 | 0.1 | 0 | 59 | - | bw | 0.2 | 10 % |
| `pf_gdn_conv` | 16.1 | 1.4 % | 335 | 126.1 | 0.168 | 376 | 0.50 | bw | 213.8 | **64 %** |
| `pf_gdn_l2norm` | 7.3 | 0.6 % | 152 | 33.6 | 0.017 | 221 | 0.11 | bw | 56.9 | 37 % |
| `pf_gdn_gate` | 0.7 | 0.1 % | 15 | 1.8 | 0 | 126 | - | bw | 3.1 | 21 % |
| `pf_gdn_A` | 15.1 | 1.3 % | 315 | 34.3 | 1.208 | 109 | 3.84 | bw | 58.2 | 19 % |
| `pf_gdn_solve` | 37.2 | 3.1 % | 775 | 50.3 | 0.128 | 65 | 0.17 | bw | 85.3 | 11 % |
| `pf_gdn_wu` | 42.2 | 3.5 % | 879 | 109.8 | 1.636 | 125 | 1.86 | bw | 186.2 | 21 % |
| `pf_gdn_A2` | 15.2 | 1.3 % | 317 | 42.3 | 1.208 | 134 | 3.81 | bw | 71.8 | 23 % |
| **`pf_gdn_scan_dpas_split`** | **63.7** | **5.4 %** | **1327** | 149.3 | 21.341 (DPAS issued; 10.67 useful) | 112 | 16.08 | bw | 253.0 | **19 %** |
| `pf_gated_head` | 16.1 | 1.4 % | 335 | 125.8 | 0.126 | 375 | 0.38 | bw | 213.3 | **64 %** |
| **GDN total** | **237.0** | **19.9 %** | | | | | | | | |

The 15,257-id prompt runs the same kernels per row (GDN does not depend on the context): 1789.1 ms of 10135.2
(17.7 %), the scan **474.9 ms = 4.7 %**, `pf_gdn_wu` 317.7 (3.1 %), `pf_gdn_solve` 282.8 (2.8 %), `pf_ab_proj` 177.8
(1.8 %); every kernel's us / launch is within 3 % of the 2048 column (its 921-row tail chunk included).

**How the bytes and FLOPs are counted (derived, C = 2048, 48 v-heads, 16 k-heads, head dim 128, 32 sub-chunks of
64):** `pf_ab_proj` reads x (C x 5120 bf16) and the 128 x 5120 bf16 weight, writes C x 128 fp32, 2 C 5120 128 FLOP;
conv reads the qkv part of the fp32 partials (C x 10240 x 4 B) and writes `xb` bf16; l2norm rewrites q and k in
place; `pf_gdn_A` / `pf_gdn_A2` read k (and q) and write a 32 x 48 x 64 x 64 fp32 tile set (25.2 MB), three of four
32 x 32 x 128 quadrants computed; the solve reads A and writes T (C(64,3) = 41,664 `fma` per head and sub-chunk);
`pf_gdn_wu` reads T, v, k, the gate and beta, writes `w` and `u` bf16 (2 x 2080 x 128 `fma` per head and sub-chunk);
the scan reads q, k, `w`, `u`, A2 and the gate, reads and writes the 48 x 128 x 128 fp32 state, writes `gdn_o` fp32
(C x 6144 x 4 B), and issues 53 `dpas.8x8` (4096 FLOP each) per sub-group per sub-chunk (192 work-groups x 16
sub-groups x 32 sub-chunks), twice the algebra's FLOPs because `S`, `D` and `A2` each run hi and lo chains; the
gated head reads z and `gdn_o` (fp32) and writes `y` bf16.

## 5. The scan's residency (`probe_gdn_scan_occ`, whole-box lock, synthetic inputs, C = 2048, 7 rounds)

The production entry over the first h heads only (grid (h, 4)). Sanity first: heads 0..5 of the h = 6 launch equal
the h = 48 launch bitwise (0 differing words of `gdn_o` and the state, all finite) - PASS.

| heads | work-groups | GPU ms | vs 48 heads |
|---:|---:|---:|---:|
| 6 | 24 | 0.306 | 0.26 |
| 12 | 48 | 0.318 | 0.27 |
| 18 | 72 | 0.401 | 0.34 |
| 24 | 96 | 0.469 | 0.39 |
| 32 | 128 | 0.912 | 0.77 |
| 36 | 144 | 0.886 | 0.74 |
| 48 | 192 | **1.191** | 1.00 |

Up to 96 work-groups (3 per Xe-core) the time barely grows: one work-group's sequential walk over 32 sub-chunks
is ~0.3 ms alone and ~0.47 ms with two neighbours on its core. At 128 it doubles: a second wave. The SLM explains
the step (**derived**): 40,448 B a work-group against 128 KB of SLM an Xe-core (the device's `local_mem_size`; the
measured step at 3 per core is consistent with it) leaves 3 resident; by threads it would be 4 (16 SIMD16 threads
of 64). The engine's launch (1.327 ms on card 0 in situ, real data) is this probe's 1.191 plus the
in-situ difference 3k recorded for its probe walk (real vs synthetic data and the neighbouring kernels' caches).

## 6. The rule, per kernel

| kernel | >= 5 % of the chunk? | < ~60 % of bound? | verdict |
|---|---|---|---|
| `pf_gdn_scan_dpas_split` | **yes, 5.4 %** (4.7 % at 15k) | **yes, 19 %** | **GO** (narrowly: the share is just over the bar) |
| `pf_gdn_wu` | no, 3.5 % | yes, 21 % | no-go |
| `pf_gdn_solve` | no, 3.1 % | yes, 11 % | no-go |
| `pf_ab_proj` | no, 2.0 % | yes, 24 % | no-go |
| `pf_gdn_conv` | no, 1.4 % | no, 64 % | no-go |
| `pf_gated_head` | no, 1.4 % | no, 64 % | no-go |
| `pf_gdn_A2`, `pf_gdn_A`, `pf_gdn_l2norm`, `pf_gdn_gate`, `pf_gdn_seed` | no, 1.3 % and below | - | no-go |

**The scan's saving (derived from section 5).** Its bound is far away (253 us against 1,327), but the bound is not
what is reachable: the kernel is a dependent chain of 32 sub-chunks per work-group, so what a remap buys is the second
wave. From the residency curve, adding a co-resident work-group to a core costs ~0.08 ms on top of ~0.47 ms at 3 per
core; all 192 resident (6 per core) would then take ~0.71 ms against 1.19, **-40 %**. Applied to the engine's
63.7 ms: **~-25 ms of a 1,190 ms chunk (~2.1 %, ~1,758 t/s at 2048) and ~-190 ms of the 15k prompt (~1.9 %)**. A
half step - 4 resident per core (SLM <= 32 KB) - leaves 192 over 128 slots and buys roughly half of that.

**Exact changes first.** A residency remap moves no rounding and reassociates no sum if each sub-group keeps its
own DPAS chains in the same order: candidates are (a) the `A2` hi / lo limbs and their VNNI packing produced by
`pf_gdn_A2` itself (the same `rne` of the same fp32 value, 10,240 B of SLM out of the scan, and the 10-block staging
loop out of every sub-chunk), (b) smaller work-groups (8 sub-groups, each owning more of the state) so that 6 fit a
core by threads and SLM. Bitwise against today's entry is the bar; a change that reorders a chain goes to the f32
judge. The parked fp32 `w` / `u` (branch `g0-fix-gdn-fp32`) adds a `w` lo chain to stage 1: the remap must leave room
for it.

**What the rule does not cover, recorded for whoever picks next.** The four WY kernels together (`pf_gdn_A`, the
solve, `pf_gdn_wu`, `pf_gdn_A2`: 109.7 ms, 9.2 % of the chunk) pass 25 MB fp32 tile sets through memory three
times; a fused A -> solve -> W/U launch per head and sub-chunk would keep A and T in SLM. The solve's 1bar variant
(bitwise, -2.3 %, docs/12-kernels.md) is still offered and not made. `pf_ab_proj` (2.0 %) could be 128 more columns
of the qkv||z GEMM. None is a build item under this rule.

## 7. Files

- `tools/probe/probe_fma_rate.cc` / `.cl` - the sustained fp32 vector `fma` rate (new; probe only).
- `tools/probe/probe_gdn_scan_occ.cc` - the production scan over the first h heads, with a bitwise sanity (new).
- `tools/probe/gdn_prefill_3m.sh` - the whole timed run under one box lock.
