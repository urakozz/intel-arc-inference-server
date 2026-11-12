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

## 4. Tasks 2, 3 and 5 on the walk (2026-11-12, branch moe-pf-phase2)

**Task 2 (`pf_gdn_front`) and Task 5 (`pf_res_norm`): bitwise.** Kernel tests on card 1: `gdn_front_test` (xb and
the ring, C 2048 / 37 / 1, pos 0 and 4093) and `pf_norm_fused_test` (residual and x, SP 0 / 1, M 2048 / 37): 0
differing words. End to end: `prefill_gate_ornith_{l0,l0-int8,l0_c16}_test` run verbose with
`B70_PF_PHASE2=old` and with the branch print identical logs (every logit / state cosine to 9 digits, every token;
only the ctest timing lines differ) - `tools/probe/pf_phase2_bitwise.sh`. The Ornith set
(`pf_phase2_gates.sh`: golden_gate_ornith, prefill_gate_ornith_* incl. c16, prefill_split_ornith_*,
ornith_prefill_* incl. the launch counts, ornith_mtp_names): 12 / 12 PASS.

Speed (card 0, interleaved, 3 rounds, unprofiled, busy box): pp4096 l0 **3,363 -> 3,439 t/s (+2.3 %)**; profiled
phase rows: norm 41.5 -> 15.8 ms (budget 12), GDN conv + l2norm 24.8 -> 20.5 ms (budget for the front <= 12: missed;
the fp32 qkv||z read is the floor, ~6 GB at 4k, and a bf16 qkv||z was measured a wash in 2026-09 - pf_gemm.cl's note).

**Task 3 (`pf_gdn_intra`): gate FAILED in both forms; not bound (opt-in `B70_PF_GDN_INTRA=hl`).**
- bf16 Q1-Q4 (D-a, 23.6 ms for the 4 stages' 80.6): l0 gates all PASS; `prefill_gate_ornith_l0-int8_test` FAILS one
  determined row (prose step 0: f32 198 by 0.16 over 271, engine 271 by 0.03; today's kernels pass).
- D2 (Q1-Q4 in fp32: `pf_gdn_intra` WU_HILO, u / w as hi / lo planes, `pf_gdn_scan_hl`), per the operator's D2:
  far closer to fp64 per layer (probe: state rel L2 1e-3 -> 5e-6, o 2-4x lower; the gate's gdn_state-vs-f32 cosines
  higher on layers 0-30), but deep layers 34-38 diverge further on `code` (L36 0.977 vs today's 0.992) and the
  token gates fail more: l0 cjk step 15 (engine misses 2 determined rows vs the bf16 golden's 1), l0-int8 prose
  steps 18 / 21 and cjk. Cost 37.4 ms intra + 31.1 ms scan (vs 80.6 + 26.4 today).
- Reading: the GDN error is no longer the dominant term; a determined-row flip follows from deep-layer routing
  (MoE near-ties) moving with any perturbation, in either direction. That is a judgement for the operator, not a
  bug to fix here (the probe's per-stage numbers, tails included at C = 42 / 2026, show no defect).

## 5. Task 6: the route GEMVs (2026-11-12)

`probe_route` (card 0, M = 2048, K = 2048, interleaved, median of 9): `pf_route.cl`'s `pf_gemv_rows` keeps
`pf_ab_proj`'s per-output arithmetic (16 K slices, k ascending inside a slice, the same `a += w * x`, the same
pairwise tree) and changes only the tiling. Every arm bitwise the control (0 differing words).

| shape | control ms | MT16 TC2 | MT32 TC1 | MT32 TC2 | MT16 TC4 | MT64 TC1 |
|---|---:|---:|---:|---:|---:|---:|
| router N 272 | 0.359 | **0.248 (0.69x)** | 0.375 | 0.382 | 0.354 | 0.416 |
| a\|\|b N 128 | 0.175 | **0.113 (0.65x)** | 0.172 | 0.144 | 0.144 | 0.192 |

MT16 TC2 is bound for both (Ornith; `B70_PF_PHASE2=old` reverts). Bitwise end to end (the verify logs below). The
gate (route + a||b <= 12 ms at 4k, today ~47) is missed: ~33 ms. D3's DPAS router (`B70_PF_ROUTE=dpas`, opt-in):
the first build measured 29.2 ms for router + moe_route against 29.3 bitwise - no gain as built - and it missed the f32 judge on l0-int8 (prefill_gate_ornith_l0-int8: cjk step 24; _c16: prose step 0); l0 passed. Not taken; the kernel was removed (reverted commit).

## 6. Landing check (rebased on main 90f1c2ae: phase 1 Tasks 1-5)

Ornith set 12 / 12 PASS; host / kernel / Flash-Next tests 15 / 15 PASS; end to end against `B70_PF_PHASE2=old`
(prefill_gate_ornith l0 / l0-int8 / l0_c16, ornith_prefill_l0, qwen4exp_prefill_synth_ours, the Flash-Next golden
prefill): identical but for timing, memory and launch-count lines. Speed (card 0, interleaved, unprofiled, busy box,
on phase 1's 66b7b234): **pp4096 l0 6,553 -> 7,035 t/s (+7.4 %, -42 ms)**.

## 7. The intra-chunk kernel on l0 (operator ruling 2026-11-12), env switches removed

`pf_gdn_intra` (bf16 Q1-Q4) is bound for MoE models on l0; l0-int8 keeps the four launches. Branch over main 9824d434:
Ornith set 12 / 12 PASS, host / kernel / Flash-Next tests 17 / 17 PASS; l0-int8 (prefill_gate_ornith_l0-int8 and
_c16) and Flash-Next (qwen4exp_prefill_synth_ours) print the same logs as main's build (bitwise unchanged). Speed
(card 0, interleaved, 3 rounds, unprofiled, busy box): **pp4096 l0 6,953 -> 7,668 t/s (+10.3 %, -54 ms)** over main.

Retried on l0-int8 after phase 1's W4A8 experts and the near-miss rule (main e31116e5): prefill_gate_ornith_l0-int8,
prefill_split_ornith_l0-int8 and ornith_prefill_l0-int8 pass; prefill_gate_ornith_l0-int8_c16 fails cjk step 15 (f32
margin 3.0 ulp, but the engine's logits cosine vs f32 0.999025 < the bf16 twin's 0.999467 - 1e-5, so the near-miss rule
does not apply). l0-int8 stays on the four launches.

## 8. The l0-int8 serving misses (part 1's W4A8 arm), checked 2026-11-12

Part 1's serving check judged 6 of 21 tool-call requests FAIL on `l0-int8` (W4A8 experts), worst speculative miss
104 ulp. **W4A8 runs on the prefill walk only** (`moe.cc`; no decode, verify or MTP-head kernel binds
`pf_moe_w4a8`), so a miss in a speculative or plain id comes from the context the prefill wrote, not from the verify.
The smallest test that shows it is `prefill_split_ornith_l0-int8_test` (long32k[0, 2600) against the f32 reference,
DIAGNOSTIC on `l0-int8`). I ran it with the experts on W4A8 and again with them forced to bf16 (a local diagnostic
build, not committed). The bf16 reference's own cosine distance to f32 is 1.0x.

| one call vs f32 (bf16 reference = 1.0x) | logits | all-rows median | fa0-3 worst | gdn L0 state |
|---|---:|---:|---:|---:|
| `l0` | x0.48 | x1.01 | 0.920 | x1.57 |
| `l0-int8`, W4A8 experts | x0.59 | x1.39 | 0.744 | x6.38 |
| `l0-int8`, bf16 experts (diagnostic) | x0.82 | x1.38 | 0.870 | x6.38 |

The deviation on long prompts is the **int8 dense linears'** (the rotated W8A8 qkv / out / o linears of spec 5): layer
0's GDN state is 6.4x the reference's distance before any expert runs, and the median moves by the same 1.38-1.39x
with or without W4A8. W4A8 adds the lower fa0-3 worst row (0.744 vs 0.870). That is no defect, only the backend's
accuracy on long prompts, which the short-prompt gates do not see. bf16 (`l0`) is the default. Nothing was changed.

## 9. Task 9: the record (main 5a78d023, card 0, profiled, busy box)

Ornith pp4096: `l0` 7,557 t/s (542 ms instrumented), `l0-int8` 7,641 (536 ms); unprofiled main is 8,080 / 8,518
(part 1's record). Phase ms at 4k (`l0` / `l0-int8`) against this plan's budget:

| phase | budget | `l0` | `l0-int8` |
|---|---:|---:|---:|
| norm | 12 / 15 | 16.5 | 16.3 |
| route (router + moe_route) | 8 | 27.9 | 27.5 |
| GDN (a\|\|b, front, gate, intra or A..A2, scan, head) | 65 | 92.9 | 151.5 |
| dense linears (+ quantiser) | 80 / 55 | 85.7 | 68.5 |
| attention | 48 | 46.9 | 45.9 |
| MoE sort + gather + GEMM + combine | 38 + 145 / 35 + 110 | 203.6 | 153.8 |
| head | - | 0.9 | 0.9 |
| sum | ~400 / ~340 | 474.6 | 464.5 |

The targets (9,500 / 11,000 t/s) are missed. The largest overruns are the MoE expert GEMM on `l0` (169 against 145),
the `l0-int8` GDN (the four intra launches, 84 ms, where `l0` runs one of 23 ms), the route (27.9 against 8) and the
scan (28 ms on both).
