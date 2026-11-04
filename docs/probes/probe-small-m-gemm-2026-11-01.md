# Probe: small-M decode GEMM (spec 28 P0, plan 28a) - 2026-11-01

Spec: `docs/superpowers/specs/2026-10-27-spec28-small-m-verify-gemm-design.md` (§6 the go rule, §9 the verdict).
Plan: `docs/superpowers/plans/2026-10-27-spec28a-small-m-probe.md`. Box queue row 68.

**Grades** (spec 28's legend): measured / derived / estimated / published by others.

**Timing grade of this record: busy-box timing (one card), to be re-checked on a quiet box.** Operator ruling
2026-11-01: the 28a timing ran on one card under `tools/gpu_lock.sh card N` while the other card and the CPU
carried other jobs (the box load is printed beside every table). Every ratio is from interleaved lists after a
warm-up, so a clock or contention shift between arms moves both sides of a pair; the absolute microseconds are
not comparable with an idle-box record. The GO / NO-GO line below is **provisional** until the re-check set (§6)
is re-run on a quiet box.

## 0. What was built

- `tools/probe/probe_small_m.cl` - the stand-in `psm` kernel (one source, compile-time knobs) and `psm_quant`
  (the int8 arm's per-row quantiser, pw4a8_quant's rule). It reads the bytes gemv / gemv_i8w read and writes
  gemv's partials (`out[(s M + m) N + n]`; the head's scaled `out[m N + n]`).
  - TYPE: bf16 DPAS (k16; int4 -> bf16 `q - 8` exact, int8 -> bf16 exact) or int DPAS on int8 rows
    (i8 x i4 k32 for int4 - the GPTQ words xor 0x88888888 are the signed B operand; i8 x i8 for the head).
  - ORIENT: `wB` (weights on B, a sub-group owns 16 columns, rows in blocks of 8 on A; M = 9..16 a second row
    block against the same B registers) or `wA` (weights on A, 8 columns per DPAS row block, two blocks per
    sub-group; tokens on B, one per lane - M <= 16 in one DPAS column block).
  - SPLITK: grid (gemv's: dimension 1 the slice; 4 sub-groups a work-group) or in-work-group (NSGN column
    sub-groups x KSL K-slices, groups interleaved across the slices, SLM reduction in slice order): 2x8 and 8x4.
  - DEQ shift (gemv's xor + arithmetic shift) or magic (the nibble OR'd into bf16 0x4300 = 128 + q, then -136
    in fp32 - exact); PF_G prefetch 1 / 2 / 3 own groups ahead (2D block prefetch for layout 0, prefetch() for the
    tiled forms); 256 GRF by the ocloc option (the probe reads the zebin's `grf_count` before timing an arm).
  - Per 64-k group: the group's DPAS chain from zero (4 x k16 / 2 x k32), then `acc += group x scale` in fp32
    (the checkpoint's g64 semantics); the head chains all K and applies the column scale at the end.
- 347 binaries (`psm_*`, tools/probe/CMakeLists.txt): per shape both types x both orientations at M = 1 / 2 / 4 /
  8 / 16, bf16 also at M = 3 / 5 / 6 / 7 (property R), the levers at M = 8, the 2x8 shape at M = 2 / 4 / 16 and
  8x4 at 16, the S sweep at M = 8; plus probe-only M = 16 controls built from gemv.cl / gemv_i8w.cl /
  gemv_bf16.cl with the production cells' defines. No scratch (spill) in any of the 347 (zeinfo).
- `tests/kernels/gemm_sm_ref.h` + `gemm_sm_ref_test` (host): the fp64 reference, gemv's fp32 twin, the i8
  quantiser, N2's checker.
- `probe_verify_m --verify8 1`: the verify lists at M = 1..8 (M = 1..4 on the engine's 4-slot MtpBuffers, 5..8
  on a probe-owned 8-slot one), unprofiled pairs and the profiled per-family table; `runtime::build_verify`
  gained an optional `ProfileEvents*` (default null, no command line changes: kernel_cmdlines +0 / -0 / ~0).

### Review Focus 2 - dequant once (ISA, measured 2026-11-01)

`IGC_ShaderDumpEnable` dumps of the gate||up shape (5120 x 34816, S 8, layout 0); instruction counts of `psm`:

| variant | M | dpas | shl | asr | shr | mov | mad |
|---|---|---|---|---|---|---|---|
| bf16 wB | 1 | 4 | 55 | 64 | 33 | 123 | 1 |
| bf16 wB | 2 | 4 | 55 | 64 | 33 | 125 | 2 |
| bf16 wB | 8 | 4 | 55 | 64 | 33 | 131 | 8 |
| bf16 wB | 16 | 8 | 55 | 64 | 33 | 171 | 16 |
| bf16 wA | 2 / 8 / 16 | 8 | 73 | 64 | 66 | 205 | 16 |

The dequant (shl / asr / shr) is the same at every M; what grows with M in wB is the activation loads' moves and
the per-row scale `mad`s (M of them a group), and the second DPAS row block at M = 16. wA's code is identical at
every M (the tokens are lanes).

## 1. The shape table (r68.shapes) - busy-box timing (one card), to be re-checked on a quiet box

`probe_small_m shapes --shape all --m 1,2,4,8,16 --rounds 5`, card 0, 2026-11-01 12:53. Box load: load average
7.1 / 6.6 / 6.3 (44 cores); the other card ran checks-1101's passkey job (pk2.sh); the CPU ran the
Qwen3.8-Flash-Next oracle (`qwen4exp_ref.py`) in a container. Every list is 40 launches (the head 8) cycling >= 72 MB
of weight copies; ctl M=1, ctl M and the arms interleaved in a rotating order after 3 warm-up executions, median of
5 (measured). Ratios are each round's own pair (the range in brackets).

**At M = 8, every shape (us per launch, measured):**

| shape (layers) | gemv M=1 | gemv M=8 | bf16 wB | bf16 wA | i8 wB | i8 wA | bf16 wB / gemv M=1 |
|---|---|---|---|---|---|---|---|
| qkv‖z 5120x16384 (48) | 81.2 | 299.0 | **83.8** | 121.6 | 102.9 | 103.6 | 1.026 (1.019-1.038) |
| out_proj 6144x5120 (48) | 30.3 | 87.9 | **30.4** | 75.4 | 37.8 | 49.1 | 1.002 (0.995-1.033) |
| q‖k‖v 5120x14336 (16) | 68.1 | 215.9 | **71.9** | 165.8 | 82.2 | 103.1 | 1.050 (1.049-1.058) |
| o_proj 6144x5120 (16) | 30.4 | 89.9 | **31.6** | 76.0 | 38.5 | 49.5 | 1.042 (1.012-1.058) |
| gate‖up 5120x34816 (64) | 170.3 | 549.3 | **193.2** | 380.1 | 202.4 | 221.4 | 1.136 (1.131-1.138) |
| down 17408x5120 (64) | 83.3 | 263.7 | **88.4** | 213.7 | 120.0 | 151.8 | 1.066 (1.057-1.069) |
| lm_head int8 5120x248320 (1) | 2206.3 | 3059.9 | 2305.3 | 2912.8 | **2270.1** | 3092.8 | 1.046 (1.039-1.047) |
| a‖b bf16 5120x128 (48, control only) | 5.0 | 14.1 | - | - | - | - | - |

The int8 arm's times include `psm_quant` before every GEMM. **bf16 wB at M = 1 / 2 / 4 / 8 / 16 against today's gemv
at M = 1** (paired medians): qkv‖z 0.999 / 0.996 / 1.002 / 1.026 / 1.142; out_proj 1.019 / 1.007 / 1.011 / 1.002 /
1.230; q‖k‖v 1.013 / 1.020 / 1.016 / 1.050 / 1.179; o_proj 1.030 / 1.013 / 1.007 / 1.042 / 1.259; gate‖up 1.021 /
1.026 / 1.080 / 1.136 / 1.283; down 1.031 / 1.029 / 1.044 / 1.066 / 1.299; the head 1.026 / 1.030 / 1.035 / 1.046 /
1.149. Today's gemv at M = 8 is 2.9-3.7x its M = 1 (the head's gemv_i8w 1.39x); at M = 16 3.8-7.7x.

**The GEMM sum of a token** (layers x median us, the head once; derived from the table):

| arm | M = 1 | M = 2 | M = 4 | M = 8 | M = 16 |
|---|---|---|---|---|---|
| today's kernels (gemv / gemv_i8w / gemv_bf16), us | 25591 | 27420 | 41990 | 79230 | 152119 |
| x today's M = 1 | 1.000 | 1.072 | 1.641 | 3.096 | 5.944 |
| **bf16 wB** (a‖b on gemv_bf16), x today's M = 1 | 1.007 | 1.017 | 1.055 | **1.100** | 1.283 |
| bf16 wA | 1.681 | 1.920 | 2.025 | 2.146 | 2.301 |
| i8 wB | 1.160 | 1.187 | 1.211 | 1.261 | 1.363 |
| i8 wA | 1.260 | 1.326 | 1.377 | 1.462 | 1.604 |

Without a‖b on either side (it stays gemv_bf16 and is in §2's non-GEMV time): bf16 wB at M = 8 = 27461 us against
today's 25351 us at M = 1 = **1.083** (derived).

**The levers at M = 8, bf16, one at a time** (`probe_small_m levers`, 12:53-12:54; lever / base arm at the same M,
paired medians, measured; the last two columns against gemv at M = 1):

| shape | wB: 2x8 wg | 8x4 wg | magic | pf 1 / 2 / 3 groups | 256 GRF | S (prod) | wA base | wA + pf1 |
|---|---|---|---|---|---|---|---|---|
| qkv‖z | 1.022 | 1.095 | 0.990 | 1.005 / 1.005 / 1.018 | **0.958** | S2 0.980 (S1) | 1.49 | 1.59 |
| out_proj | 1.234 | 1.156 | 0.988 | 1.004 / 1.025 / 1.036 | 1.474 | S16 1.074 (S4) | 2.49 | 1.18 |
| q‖k‖v | 1.090 | 1.026 | 0.985 | 1.000 / 1.010 / 1.031 | 1.200 | S4 1.059 (S2) | 2.43 | 1.13 |
| o_proj | 1.211 | 1.087 | 0.980 | 0.977 / 0.995 / 1.028 | 1.460 | S16 1.042 (S4) | 2.50 | 1.17 |
| gate‖up | **0.942** | 1.023 | 0.984 | 1.010 / 1.015 / 1.024 | 1.052 | S4 0.988 (S8) | 2.24 | 1.23 |
| down | 1.210 | 1.072 | 0.989 | 0.995 / 0.980 / 0.986 | 1.527 | S16 1.059 (S4) | 2.56 | 1.17 |
| lm_head | **0.952** | 0.951 | 0.999 | 1.020 / 1.032 / 1.043 | 1.008 | - | 1.32 | 1.18 |

- **Orientation: wB.** wA costs 2.2-2.6x gemv's M = 1 on the int4 shapes; prefetch halves it (its weight rows are
  per-lane 64-byte broadcast loads - the probe's way of putting our layouts on the A operand without a repack), but
  it stays 1.13-1.59x where wB is 1.00-1.14x. oneDNN's wA wins with a layout made for it; ours is gemv's.
- **Type: bf16.** i8 x i4 is never >= 10 % faster (the sum at M = 8: 1.261 against 1.100) and fails N2 against the
  bf16 rows (§3); spec 28 §2's rule keeps bf16.
- **S: the production S everywhere** (no S >= 5 % faster at M = 8: qkv‖z S2 2.0 %, gate‖up S4 1.2 %).
- **The in-work-group split-K** pays on the long-N shapes only (gate‖up 0.94; the head 0.95, and 0.89-0.95 at every
  M) and costs 9-28 % on the K-short ones; one shape for every M (property R) at M = 16: 2x8 0.87 on qkv‖z and 0.89
  on the head, 1.09-1.20 on the rest. A per-shape choice, not a global one.
- **Magic dequant: 1-2 % everywhere in wB** (0.980-0.999) - inside this session's noise. 256 GRF helps qkv‖z only.
- Levers combined per shape (estimated: the products of the single-lever ratios with the shape table's base,
  never measured together): qkv‖z 256 GRF, gate‖up and the head 2x8, magic on the rest -> 26.30 ms for the M = 8
  sum (27.46 measured base).

## 2. The verify step's attribution (r68.attrib) - busy-box timing (one card)

`probe_verify_m --verify8 1 --depth 4096 / 32768 --replays 32 --rounds 3` and `probe_mtp_steps <snap> 4096 32 3
int8 off 16384 8` (r29.cost's arm - it had not run), card 0, 12:54-13:00, load average 4.1-7.5, the other card on
checks-1101's jobs. The verify lists: int8 head, attention v2, bf16 KV.

**The unprofiled verify step today, in plain steps (verify M = 1), measured:**

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| depth 4096, probe_verify_m 1v (ms) | 30.50 | 35.37 | 46.24 | 52.29 | 61.24 | 71.87 | 79.89 | 96.57 |
| x M = 1 (paired) | 1.000 | 1.144 | 1.461 | 1.697 | 2.005 | 2.361 | 2.625 | **3.170** |
| depth 4096, probe_mtp_steps pairs (x M = 1) | 1 | 1.158 | 1.514 | 1.716 | 2.005 | 2.375 | 2.629 | **3.167** |
| depth 32768, 1v (x M = 1; M = 1 is 34.65 ms) | 1 | 1.219 | 1.592 | 1.831 | 2.392 | 3.151 | 3.612 | 4.277 |

verify(8) is 3.17 plain steps, measured - not the 2.62 docs/probes/probe-dflash-2026-10-08.md §3 extrapolated.

**Profiled per-family time at depth 4096** (ms/step; each launch's event adds a flush - shares, not times):

| group | M=1 | M=2 | M=3 | M=4 | M=5 | M=6 | M=7 | M=8 |
|---|---|---|---|---|---|---|---|---|
| int4 GEMV (gemv) | 23.76 | 26.81 | 36.37 | 41.57 | 48.96 | 56.47 | 62.88 | 71.90 |
| int8 head (gemv_i8w) | 2.21 | 2.25 | 2.28 | 2.27 | 2.32 | 2.43 | 2.87 | 3.17 |
| attention (attn_decode_v2 + attn_prep + attn_reduce_v2) | 0.90 | 1.48 | 1.96 | 2.45 | 3.07 | 5.40 | 6.45 | **12.28** |
| gdn_step | 0.78 | 1.17 | 1.51 | 1.91 | 2.31 | 2.69 | 3.10 | 3.38 |
| everything else (norms, silu, a‖b and the MTP head's bf16 GEMVs, argmax, embed) | 2.05 | 2.32 | 2.40 | 2.60 | 2.68 | 3.01 | 3.09 | 3.15 |
| sum | 29.70 | 34.03 | 44.52 | 50.79 | 59.34 | 69.99 | 78.38 | 93.87 |
| non-GEMV share x the unprofiled step (ms, derived) | 3.83 | 5.17 | 6.10 | 7.16 | 8.32 | 11.40 | 12.88 | **19.35** |

At depth 32768 the attention is 5.07 ms at M = 1 and 66.40 at M = 8 (non-GEMV derived 8.08 -> 73.81 ms of a 148.36 ms
step). attn_decode_v2 doubles between M = 7 and 8 at depth 4096 (6.26 -> 12.09 ms; the B70_VERIFY_M8 block's comment
expected a possible spill at 48 (head, row) pairs a lane) - spec 28c's subject.

## 3. Numerics (r68.check, r68.numerics)

**r68.check** (`probe_small_m check --shape all`, card 1, 12:44-12:46; random weights and rows): 398 variant runs
(every binary of every shape; out_proj and o_proj share theirs), **0 failures, 0 missing binaries**. bf16 arms
against the fp64 reference: max |err| / (2^-16 sum|w x|) = 0.0034, min cosine 1.000000000 (N2: <= 1 and >=
0.9999999); the controls 0.0039. The int arm: the device quantiser bitwise the host's; the kernel against the fp64 of
its own quantised rows 0.002 of the bound; against the bf16 rows up to 58.7x the bound, min cosine 0.99995. Property
R: every row of every multi-M configuration bitwise across M = 2..8 / 16 (grid and in-work-group, both orientations,
both types) and with the other rows random (4 trials each). The ISA counts are §0's. After the branch review found
that N2's checker let a NaN output through (NaN compares false), the checker was fixed (a non-finite output fails;
gemm_sm_ref_test holds it) and both r68.check and r68.numerics were re-run on card 0 at 13:20-13:33: unchanged,
398 runs 0 failures, numerics PASS.

**r68.numerics** (`probe_small_m numerics <snap> --shape all --trials 1000`, card 0, 13:00-13:04): layer 62's qkv‖z
and out_proj, layer 63's q‖k‖v / o_proj / gate‖up / down and the int8 head, as loaded; 32 real rows (two prompts of
long32k.ids x 16 greedy tokens: the final-norm hidden for the K 5120 shapes and the head, layer 63's attention output
for K 6144, and layer 63's silu(gate) x up of those rows, computed on the host, for down) and random rows; M = 1 / 2
/ 4 / 8 / 16.

| arm | rows | N2 vs fp64: max ratio | min cosine | N2 vs gemv at the same M: max ratio |
|---|---|---|---|---|
| bf16 wB / wA | real | 0.0125 | 1.000000000 | 0.0261 |
| bf16 wB / wA | random | 0.0041 | 1.000000000 | 0.0063 |
| gemv (control) | real / random | 0.0259 / 0.0030 | 1.000000000 | - |
| i8 wB / wA | real | 2150 (down; 312-397 elsewhere at M = 8) | 0.99315 (down; >= 0.998 elsewhere) | - |
| i8 wB / wA | random | 64.7 | 0.99992 | - |

- **N2 passes** for both bf16 orientations on every shape, real and random rows, against the fp64 reference and
  against gemv (80x and 38x under the bound).
- **N3 (property R) passes:** 1000 trials a shape and arm (M 16 / 8 / 4 / 2 in turn, a real or random row kept at a
  random index among random rows, against itself at M = 16 among zero rows): 0 differing row-slices of 96,000
  compared (bf16 and i8, wB and wA, every slice of S).
- **The int arm** computes its own definition (0.0027 of the bound against its quantised rows' fp64), but unrotated
  per-row int8 rows lose up to cosine 0.9931 (down, real rows). The logit-level error on the four prompts
  (docs/probes/probe-w4a8-2026-09-23.md §15.4, a CPU oracle run) was not run (the CPU model lock was held); the type rule
  needs no more - i8 is not faster.

## 4. The provisional verdict (spec 28 §6) - busy-box timing, provisional

- (a) **the GEMM sum at M = 8 = 1.083-1.100x** today's M = 1 sum (bf16 wB, production S, no lever): passes (<= 1.25).
- (b) **derived verify(8)** = §2's non-GEMV time at M = 8 (19.35 ms) + (a)'s sum without a‖b (27.46 ms) = 46.81 ms =
  **1.535 plain steps** (30.50 ms): in the 1.4-1.6 band. With the levers combined (estimated, 26.30 ms): 1.50.
  The two terms are not measured the same way: the GEMM sum is stand-alone lists, the non-GEMV time is in-list. At
  M = 1 the same sum gives 25.35 + 3.83 = 29.18 ms against the measured 30.50 - today's GEMVs cost 5.2 % more in
  the list (26.67 ms derived) than alone. Charging the stand-in the same 5.2 %: (19.35 + 1.052 x 27.46) / 30.50 =
  **1.58** (derived) - still in the band, near its top. Both are given; the quiet-box re-check should settle which.
- (c) N2 and N3 pass (§3).

**Provisional line: the operator's call.** (a) and (c) pass; (b) is 1.54-1.58. What holds (b) up is not the GEMM: the
non-GEMV time at M = 8 is 19.35 of the 46.81 ms, and 12.3 ms of it (profiled) is attention, whose M = 7 -> 8 step
doubles. At the attention's M = 1..7 slope (0.93 ms a row, profiled) verify(8) would be ~1.37 plain steps (estimated) -
spec 28c's lever, not this spec's. Against today's measured verify(8) of 3.17 plain steps, the derived 1.54 is 2.07x
fewer steps per verify.

Recorded, not gating: wB over wA; bf16 over i8; the production S; magic dequant 1-2 %, the 2x8 work-group on gate‖up
and the head 5-6 %, 256 GRF on qkv‖z 4 %; gemm_sm at M = 1 is 0.7 % slower over a token than gemv (spec 28 §8
question 1 stands: gemv keeps M = 1); depth 32768's non-GEMV share is 50 % at M = 8.

## 5. Open

- The quiet-box re-check (§6) replaces the provisional line.
- The int arm's logit-level error (CPU oracle): not needed for the type rule (i8 is slower), recorded as not run.
- 28b's per-shape levers (2x8 on gate‖up and the head, 256 GRF on qkv‖z, magic) need one combined measurement.
  Property R is shown (§3) for the base arms (grid split-K, shift dequant, both types and orientations) and for the
  2x8 / 8x4 in-work-group shapes across M; magic dequant, prefetch, 256 GRF and the S sweep were built at M = 8
  only, so their row independence is untested here (expected by construction - nothing in them depends on M - but
  28b must test it). A per-shape in-work-group shape must stay one shape for every M >= 2.
- The check's references cover all columns up to N = 6144 and three 512-wide windows (start, middle, end) above
  it (gate‖up 4.4 % of its columns, the head 0.6 %): "0 failures" is on those windows.

## 6. The quiet-box re-check set

The points that decide the line and the ones nearest the 3 % noise line, from the box tree (`~/b70/trees/spec28a`,
or main's `build` / `build-extra-m` once merged), both cards free of DRM holders:

```
SNAP=$HOME/.cache/huggingface/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d
# 1. (a): every shape, gemv vs bf16 wB at M = 1 and 8 (gate‖up is 45 % of the sum; then down, qkv‖z, the head)   ~1 min
tools/gpu_lock.sh box -- env ZE_AFFINITY_MASK=0 build/tools/probe/probe_small_m shapes --shape all --m 1,8 --type bf16 --orient wB --rounds 5
# 2. (b): the verify step at M = 1..8 and the non-GEMV share at depth 4096                                         ~2 min
tools/gpu_lock.sh box -- env ZE_AFFINITY_MASK=0 build/tools/probe/probe_verify_m $SNAP --verify8 1 --depth 4096 --replays 32 --rounds 3
# 3. the noise-line levers: 2x8 on gate‖up / the head (0.94 / 0.95), 256 GRF and S2 on qkv‖z (0.96 / 0.98),
#    magic everywhere (0.98-0.99)                                                                                     ~1 min
tools/gpu_lock.sh box -- env ZE_AFFINITY_MASK=0 build/tools/probe/probe_small_m levers --shape gateup,head,qkvz --orient wB --rounds 5
tools/gpu_lock.sh box -- env ZE_AFFINITY_MASK=0 build/tools/probe/probe_small_m shapes --shape all --m 8 --type bf16 --orient wB --deq magic --rounds 5
```

`probe_verify_m` needs a build with `-DB70_DECODE_EXTRA_M=ON` (the spec28a box tree's `build` has it; box_validate's
r68.build makes `build-extra-m`). The full 2026-11-01 session (shapes, levers, verify at 4096 / 32768, r29.cost's
arm, numerics) took 11 minutes on one card.
