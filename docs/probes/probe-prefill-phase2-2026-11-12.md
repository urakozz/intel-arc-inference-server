# MoE-model prefill phase 2, Task 1 - the per-launch trace and the GDN intra-chunk arms (2026-11-12)

Plan: `docs/superpowers/plans/2026-11-12-moe-model-prefill-phase2.md` Task 1. Branch `moe-pf-phase2` (main 49171010 +
the probe commits). Ornith 1.5 35B-A3B int4, one B70. Box row 84 (`r84.trace`, `r84.dump`, `r84.probe`, opt-in).
Logs: `~/b70/logs/moe-prefill-phase2/` on the box.

## Gate (the plan's, pre-registered there before any of these runs)

- D-arms: every stage's max relative error and the state / o cosine against fp64 within 2x the control's own, on all
  three layers (Ornith; Flash-Next waits for Task 8 - card 1 holds the operator's quantisation); an arm over it fails.
- Speed: the best passing D-arm's A + T + W/U + A2 <= 0.35 x today's per layer and chunk (today 1.34 ms at C = 2048).
- The trace accounts for >= 95 % of l0-int8's 114 ms "norm" row by named launches.
- Dense: Task 7 runs only if a dense arm saves >= 10 % of the dense phase at M = 4096.

## 1. The trace (`B70_PREFILL_PROFILE=2`, pp4096, card 0, round 2 of 2, busy box)

The 114.3 ms "norm" row of l0-int8 is **73.9 ms of `pf_colmax_rot`** (120 launches: 80 dense linears + 40 MoE
gate||up arrays) - the one-time rotated column-scale pass `Engine::prepare_prefill` runs at load, OUTSIDE the
measured call. It ended with a bare `cx.wait()`, so its timestamps waited in the Context until the walk's first
`profile_wait` (layer 0's norm) took them. The rest is the same 41 ms as l0 (`pf_res_fold` 25.9 + `pf_norm_finish`
14.4 + embed 0.04). Named launches: 100 % of the row. **Gate: PASS.** Fixed in `Int8State::compute_scales` (its own
`setup` bucket; unprofiled it is the same `cx.wait()`), so l0-int8's real kernel sum at 4k is ~1024 ms, not 1097,
and Task 5's int8 budget line (114 -> 15) is really 41 -> 15.

| phase | entry | launches | l0 ms | l0-int8 ms |
|---|---|---:|---:|---:|
| norm | pf_res_fold | 161 | 26.81 | 25.90 |
| norm | pf_norm_finish | 161 | 14.54 | 14.39 |
| norm | pf_colmax_rot (setup, mis-bucketed) | 120 | - | 73.93 |
| dense | pf_dequant_slab / pf_gemm | 1060 / 1060 | 12.64 / 76.16 | - |
| dense | pf_quant_had / pf_requant_rot / pf_gemm_i8 | 160 / 1060 / 1060 | - | 5.52 / 16.16 / 50.30 |
| GDN | pf_ab_proj (a\|\|b) | 60 | 10.85 | 10.84 |
| GDN | pf_gdn_seed / conv / l2norm / gate | 60 each | 0.10 / 16.28 / 8.44 / 0.51 | 0.10 / 15.93 / 8.24 / 0.52 |
| GDN | pf_gdn_A / solve / wu / A2_hilo | 60 each | 10.88 / 26.49 / 32.01 / 11.61 | 10.69 / 26.16 / 31.54 / 11.37 |
| GDN | pf_gdn_scan_dpas_split / pf_gated_head | 60 each | 26.34 / 12.52 | 26.53 / 12.46 |
| attention | pf_attn_prep / pf_flash_attn / pf_attn_gate | 20 each | 4.95 / 39.19 / 4.00 | 4.89 / 37.66 / 4.07 |
| MoE route | pf_ab_proj (router, N 272) / moe_route | 80 each | 29.90 / 5.77 | 29.35 / 5.65 |
| MoE | sort / gather / weights / GEMM / combine | | 123.1 / 14.2 / 390.7 / 277.0 / 14.0 | 121.5 / 7.1 / 331.5 / 200.7 / 14.1 |

The small kernels need no separate probe: the trace times each alone on the card. Bandwidth floors at ~500 GB/s
(4k, all calls): the norm pair moves ~6.7 GB (fp32 partials on the 80 mixer folds, bf16 rows) -> ~13 ms against
41; the router GEMV reads x [C][2048] bf16 once per layer (8.4 MB x 80 = 0.7 GB, ~1.4 ms) against 29.9; a||b the
same x (~0.5 GB, ~1 ms) against 10.9. `probe_small_kernels.cc` was not written for that reason.

## 2. The GDN intra-chunk arms on real inputs (`probe_gdn_intra`)

Inputs: one pp4096 prefill's GDN calls 0 / 9 / 21 (layers 0 / 12 / 28, chunk 0, C = 2048), `B70_PF_GDN_DUMP`.
Control: `pf_gdn_A`, `pf_gdn_solve`, `pf_gdn_wu`, `pf_gdn_A2_hilo`. **D-a**: `pf_gdn_intra` (src/kernels/prefill/
pf_gdn_intra.cl) - A and A2 by bf16 DPAS (exact operands), T in 16 x 16 fp32 blocks (diagonal blocks by
pf_gdn_solve's own chain, off-diagonal by T_ij = -T_ii SUM A_ik T_kj), W / U by DPAS with T as hi / lo bf16 pairs,
one work-group per (head, sub-chunk), 32 KB SLM. D-b (A2 apart) was not needed: D-a's SLM is 32 KB, 4 work-groups an
Xe-core. D-c (tf32 DPAS) not built: the hi / lo pair already meets the band.

Errors against an fp64 walk of the same inputs (exact algebra; both arms carry Q1-Q4, which dominate w / u / o):
rel L2 = |x - ref| / |ref|, max = max |x - ref| / max |ref|; o and state as rel L2 and cosine distance after today's
scan from a zero state. Card 1.

| layer | stage | control rel L2 | control max / cos | D-a rel L2 | D-a max / cos | ratio |
|---|---|---:|---:|---:|---:|---|
| 0 | A | 7.335e-08 | 1.861e-07 | 7.364e-08 | 2.105e-07 | 1.00, 1.13 |
| 0 | T | 7.140e-08 | 3.505e-07 | 7.120e-08 | 4.041e-07 | 1.00, 1.15 |
| 0 | A2 | 2.456e-06 | 6.127e-06 | 2.456e-06 | 6.127e-06 | 1.00, 1.00 |
| 0 | w / u | 3.256e-03 / 2.492e-03 | 4.344e-03 / 4.642e-03 | same | same | 1.00 |
| 0 | o / state (cos) | 9.427e-04 / 5.579e-03 | 4.373e-07 / 1.556e-05 | 9.432e-04 / 5.578e-03 | 4.378e-07 / 1.556e-05 | 1.00 |
| 12 | A / T | 7.902e-08 / 1.076e-07 | 2.476e-07 / 3.760e-07 | 7.177e-08 / 9.471e-08 | 2.091e-07 / 3.167e-07 | 0.91 / 0.88 |
| 12 | A2, w, u | | | | | 1.00 |
| 12 | o / state (cos) | 1.555e-03 / 1.274e-03 | 1.203e-06 / 7.491e-07 | 1.558e-03 / 1.273e-03 | 1.208e-06 / 7.477e-07 | 1.00 |
| 28 | A / T | 7.700e-08 / 7.649e-08 | 2.304e-07 / 3.097e-07 | 7.310e-08 / 7.258e-08 | 2.014e-07 / 2.925e-07 | 0.95 / 0.95 |
| 28 | A2, w, u | | | | | 1.00 |
| 28 | o / state (cos) | 6.689e-04 / 1.453e-03 | 1.255e-07 / 9.612e-07 | 6.698e-04 / 1.453e-03 | 1.258e-07 / 9.613e-07 | 1.00 |

**Numerics gate: PASS** (worst ratio 1.15, T's max on layer 0). The production binary's w equals the debug
binary's word for word.

Time (card 0, busy box: phase 1's agent shares the box; interleaved pairs after 3 warm-ups, median of 9, L0
global timestamps around the launches):

| layer | C | control ms (4 launches) | D-a ms (1 launch) | ratio |
|---|---:|---:|---:|---:|
| 0 | 2048 | 1.366 | 0.396 | 0.290 |
| 12 | 2048 | 1.722 | 0.504 | 0.293 |
| 28 | 2048 | 1.727 | 0.512 | 0.296 |
| 0 | 4096 | 2.639 | 0.709 | 0.269 |
| 12 | 4096 | 2.670 | 0.712 | 0.267 |
| 28 | 4096 | 2.680 | 0.718 | 0.268 |

**Speed gate: PASS** (0.27-0.30 <= 0.35). At the profile's 1.34 ms a call that is ~0.39-0.40 ms, ~24 ms over the
60 calls of a 4k prefill (budget 28).

## 3. Not run

- Flash-Next's dumps (`--layers 18`): card 1 holds the operator's quantisation; Task 8 runs them.
- The dense arm (Task 7): not probed. The dense GEMMs run at 75 % (bf16) / 57 % (int8) of peak (the profile); the
  arm is low priority in the plan and Task 7 stays closed unless a later probe shows >= 10 %.
