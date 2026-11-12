# MoE-model prefill, phase 2 - the GDN prompt pass, the small kernels, the dense linears (Ornith, then Flash-Next)

**Status (2026-11-12): IN PROGRESS** - Task 1 done; Tasks 2 and 5 done (bitwise, +2.3 % pp4096 l0); Task 3 STOPPED
(its gate failed in both forms, opt-in `B70_PF_GDN_INTRA=hl`, waits for the operator); Task 6 done bitwise
(+7.4 % with Tasks 2 and 5 on phase 1's walk); D3 tried and not taken. Record: docs/probes/probe-prefill-phase2-2026-11-12.md. Phase 1 is `2026-11-12-ornith-moe-prefill.md` (cf5ed009, being implemented): the
MoE path's sort, fused int4 grouped GEMM, l0-int8 experts as W4A8 and the 4096 chunk. This plan is everything phase 1
leaves out. Evidence: `docs/probes/ornith-prefill-profile-2026-11-12.md` (card 0); the scout's code read of vLLM's
XPU kernels (vllm-xpu-kernels @1f31a8e: `csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp`,
`chunk_causal_conv1d_tiled_xe2.hpp`, `csrc/layernorm.cpp`, `csrc/moe/topk.cpp`; no timings in that repo - every vLLM
ms is derived); our GDN history in `pf_gdn_scan.cl`'s header (the single-bf16 DPAS scan cost a token on 2026-09-05;
the hi / lo split-bf16 scan passed and is the default since 2026-09-23). Box row **84** (new, Task 1 adds it).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One
> implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox
> (`- [ ]`) syntax. **Commit after every step** (signed, no trailers; a step that does not build or pass yet goes in as a
> `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop. Box jobs: every GPU job
> through `tools/gpu_lock.sh card 0` (speed) or `card <N>` (correctness), long jobs detached (`nohup setsid`, logs
> under `~/b70/logs/moe-pf-phase2/`), checked at most every 30 min with one blocking wait.

**Targets (operator, 2026-11-12):** Ornith prefill on one card at pp4096: **>= 11,000 t/s on `l0-int8`** (~370 ms
per 4096 ids) and **>= 9,500 t/s on `l0`** (~430 ms). The 16k numbers are recorded, not gated. Decode unchanged.

## The budget (kernel ms at 4k; the wall is ~30 ms above the kernel sum, measured 1218 vs 1188 today)

| phase | today l0 / int8 | after phase 1 (its derived numbers) | budget for the targets l0 / int8 | owner |
|---|---:|---:|---:|---|
| MoE GEMMs + expert weights | 668 / 531 | ~260 / ~170 | **145 / 110** | phase 1 |
| MoE sort, gather + combine | 151 / 141 | ~38 | 38 / 35 | phase 1 |
| MoE route (router GEMV + moe_route) | 36 / 35 | 36 | 8 | this plan, Task 6 |
| GDN (incl. a\|\|b, gated head) | 156 / 154 | 156 | **65** | this plan, Tasks 2-4 |
| norms (+ l0-int8's quantisers) | 41 / 114 | 41 / 114 | 12 / 15 | this plan, Task 5 |
| dense linears | 89 / 72 | ~83 / ~66 (the 4096 chunk halves the dequant) | 80 / 55 | this plan, Task 7 (if probed) |
| attention, head | 48 / 48 | 48 | 48 | - |
| **sum** | 1188 / 1097 | ~660 / ~550 | **~400 / ~340** | |

**DERIVED, not measured.** What this plan can deliver on its own: ~-155 ms on l0 and ~-225 ms on l0-int8 (table
below). With phase 1 at its own derived numbers that lands at ~500 / ~355 ms wall, ~8,200 / ~11,500 t/s: **the l0-int8
target is in reach, the l0 target needs phase 1's GEMM at ~145 ms** - about vLLM's derived 35-45 % of peak with
in-register dequant, against phase 1's own estimate of ~260 ms. Task 9 reports the measured split against this table, so
the gap, if there is one, has a name.

| task | what | 4k saving l0 / int8 (derived) | numerics |
|---|---|---:|---|
| 2 | GDN front: seed + conv + SiLU + l2norm + gate in one launch; qkv\|\|z written bf16 | ~20 / ~20 | bitwise |
| 3 | GDN intra-chunk: A, T = (I + A)^-1, W / U, A2 in one DPAS kernel per (head, chunk) | ~62 / ~62 | judged |
| 4 | GDN scan + gated head (only what Task 1 shows) | ~9 / ~9 | bitwise or judged, per arm |
| 5 | residual + RMSNorm in one launch; l0-int8's quantiser folded into the norm | ~29 / ~99 | bitwise |
| 6 | route: router + a\|\|b GEMVs re-tiled, moe_route fused in | ~35 / ~35 | bitwise first; judged arm only by D3 |
| 7 | dense linears, only if the probe shows >= 10 % of their time | ~0-20 / ~0-15 | judged |

**Order and why:** measure first (Task 1): the trace that explains l0-int8's 114 ms norm bucket, and the GDN DPAS
arms' error against fp64 before their speed - DPAS has cost this model family a token before. Then the bitwise GDN front
(Task 2: safe, and it frees the bf16 qkv\|\|z that Task 3 reads), the judged intra-chunk kernel (Task 3, the big one),
the scan (Task 4), the norms (Task 5), the route (Task 6), the dense linears (Task 7, low priority), Flash-Next (Task 8),
the record (Task 9).

**Stop rules:**
- Task 3: if no DPAS arm in Task 1 stays within the error band of today's vector kernels (the gate below), Task 3 does
  not start and the plan reports the GDN budget as missed.
- Any bitwise gate that fails stops its task until the first differing value is explained.

## Dependencies

- Main at or after cf5ed009. Branch `moe-pf-phase2`, own worktree. Phase 1 runs in parallel: Tasks 1-5 and 7-8 touch
  no file phase 1 owns (`moe.cc`, `pf_moe*.cl`, `buffer_sizes.cc`'s MoE layout); **Task 6 touches `moe.cc` and starts
  after phase 1's Task 3 is on main** (rebase, not merge). Task 5's quantiser fold for the post-attention norm takes
  phase 1's W4A8 activation quantiser as it lands (its Task 5).
- Box data: the Ornith int4 checkpoint and its f32 references (`oracle_ornith_f32`, `oracle_ornith_f32_pfi8`, the
  long32k reference prefill_split_ornith uses); Flash-Next's synthetic checkpoints and Intel's checkpoint at
  `--layers 18`. Card 1 holds the operator's Flash-Next quantisation over the weekend: until it ends, card 0 only.

## Global constraints

- **Additive kernels.** New sources: `src/kernels/prefill/pf_gdn_front.cl` (Task 2), `pf_gdn_intra.cl` (Task 3),
  `pf_norm_fused.cl` (Task 5); Task 6 adds entries to a new `pf_route.cl`. Every existing binary and command line stays
  byte for byte until Task 9 (`tools/kernel_cmdlines`: + only; G0's `g0.sha` shows every pre-existing binary
  identical).
- **Who moves.** Ornith binds the new kernels. Flash-Next binds them in Task 8 (its GDN is Qwen3.8's shape: 16 k / 48 v
  heads, dim 128 - it binds Qwen3.8's `pf_gdn_*` binaries today). **Qwen3.8 stays on today's kernels, bitwise under
  G0**, unless the operator switches it (D1) with its own gates. **K2, Kolibri and Bolt are untouched** (host and
  binaries; G0).
- **A/B switch:** `B70_PF_PHASE2=old` (env, tests and probes only) puts Ornith / Flash-Next back on today's kernels. No
  user-facing flag. Task 9 deletes the losers and the variable. **Done early (operator 2026-11-12):** the variable,
  `B70_PF_GDN_INTRA` and the old paths they selected are gone; A/B runs compare a control build against the branch's
  (`tools/probe/pf_phase2_ab.sh`, `pf_phase2_bitwise.sh`).
- **Numerics:** bitwise tasks are checked by the G0 mechanism extended for the run (`G0_BITWISE_RE` plus
  `prefill_gate_ornith_{l0,l0-int8}_test`, `_c16`, `ornith_prefill_{l0,l0-int8}_test`, `--baseline main`). Judged tasks
  use the f32 judge with the 1-bf16-ulp near-tie rule (memory: moe-gates-f32-judge): the engine's differing MoE sets and
  token misses no more than the bf16 golden's against f32, both counts printed; the near-miss rule for the split gates
  (logits cosine distance <= 2x the bf16 twin's, extra misses within 8 bf16 ulp); l0-int8 against the plain f32
  reference. **GDN state stays fp32**; no fp32 quantity becomes a single bf16 DPAS operand - it is carried as a hi / lo
  bf16 pair (`hi = rne(x)`, `lo = rne(x - hi)`), the form the scan has used since 2026-09-23. Deterministic: no
  atomics, fixed reduction orders, replay bitwise.
- **Timing:** interleaved pairs after a warm-up, median of rounds, card 0 only; each number labelled "quiet box" or
  "busy box"; probe ms are L0 kernel timestamps.
- **Launch counts:** `kGdnChunkLaunches` (10 today) and `moe_chunk_launches` stay the homes of the counts; the tests
  asserting 2061 / 2021 and Flash-Next's `prefill_chunk_launches` read them.

---

### Task 1: the trace and the probes (Mac, then box)

**Files:**
- Modify `src/runtime/prefill/profile.{h,cc}`: `B70_PREFILL_PROFILE=2` prints every launch's entry name, phase and
  device ms (today level 1 sums per phase; a launch with no `profile_wait` of its own lands in the next phase's bucket,
  which is the suspected cause of l0-int8's 114 ms "norm" row with 120 more launches).
- Create `tools/probe/probe_gdn_intra.cc` + `.cl`: one GDN layer's intra-chunk stages on REAL inputs (q / k / v, g,
  beta dumped from an Ornith 4096-id prefill by `ornith_prefill_test --dump-gdn DIR`, layers 0, 12, 28; Flash-Next
  from `--layers 18`), C = 2048 and 4096:

  | arm | what |
  |---|---|
  | control | today's `pf_gdn_A`, `pf_gdn_solve`, `pf_gdn_wu`, `pf_gdn_A2_hilo` |
  | D-a | one work-group per (head, chunk): A = K K^T by bf16 DPAS (k is bf16 already: exact operands, only the sum order moves), x beta x exp(gc_i - gc_j) in fp32; T by 16 x 16 DPAS blocks (diagonal blocks by forward substitution in fp32 registers, off-diagonal blocks T_ij = -T_ii sum A_ik T_kj by DPAS with A and T as hi / lo pairs); W / U = T (kb / vb) with T as hi / lo (2 DPAS a product); A2 = Q K^T by bf16 DPAS, Q_SCALE applied to the fp32 result; everything in SLM, one launch |
  | D-b | D-a with A2 left in its own launch (today's `pf_gdn_A2_hilo`), if D-a's SLM footprint cuts occupancy |
  | D-c | D-a with T as tf32 DPAS, only if the B70's compiler exposes a tf32 DPAS builtin (checked, not assumed) |

  Per arm: the error of A, T, w, u, A2 and of one scan's state and o (today's scan run on each arm's outputs) against
  an fp64 host walk of the same inputs (`tests/prefill/gdn_chunk_ref.h`), beside the control's own error; then ms.
- Create `tools/probe/probe_small_kernels.cc`: the route (router GEMV, moe_route), a||b GEMV, res_fold + norm_finish
  and the l0-int8 quantiser alone on the card at C = 2048 / 4096: ms each, against their bandwidth floor.
- Create (low priority) a dense-linear arm in `tools/probe/probe_pf_gemm.cc`: qkv||z 2048 x 12288 and out_proj 4096 x
  2048 at M = 4096, today's slab + `pf_gemm` and `pf_gemm_i8` against a 128 x 256 work-group tile with 2D prefetch
  distance 3 (the w4a16 policy the scout read), bitwise checked against the control where the arithmetic allows.
- Modify `tools/box_validate/stages.sh`: row 84, stages `r84.trace`, `r84.dump`, `r84.probe` (opt-in, timed).

- [x] **Step 1 (Mac):** the profile level, the dump mode, the probes, row 84; `tools/box_validate.sh --dry-run --only
  r84` prints the commands; host tests green. Commit `probes: the per-launch prefill trace, the GDN intra-chunk DPAS
  arms on real inputs, the small kernels (phase 2 Task 1)`.
- [x] **Step 2 (box, card 0):** `r84.trace` (4k, l0 and l0-int8), `r84.dump`, `r84.probe`. Record
  `docs/probes/probe-prefill-phase2-2026-11-xx.md`: the per-launch table (what the 114 ms is), each arm's error and ms.
  - **Gate (pre-registered, written into the record before the run):**
    - D-arms: every stage's max relative error and the state / o cosine against fp64 within 2x the control's own (the
      split-gate rule's factor), on all three layers and both models; an arm over it is a failed arm.
    - Speed: the best passing D-arm's A + T + W/U + A2 <= 0.35 x today's (80.6 ms at 4k l0 -> <= 28 ms, scaled per
      layer and chunk: today 1.34 ms).
    - The trace accounts for >= 95 % of the 114 ms by named launches.
    - Dense: Task 7 runs only if a dense arm saves >= 10 % of the dense phase at M = 4096 (profile: 75 % of the bf16
      peak, 57 % of int8).

### Task 2: the GDN front in one launch (Mac, then box) - BITWISE

Today: `pf_gdn_seed`, `pf_gdn_conv`, `pf_gdn_l2norm`, `pf_gdn_gate` (4 launches, 16.3 + 8.4 + ~1-2 ms at 4k), reading
the qkv||z linear's fp32 output [C][12288] (Ornith) that the conv rounds to bf16 at one point (gdn_step.cl:253's).
vLLM's conv does conv + split + SiLU + l2norm in one kernel; ours can too, with today's arithmetic.

**Files:** create `src/kernels/prefill/pf_gdn_front.cl` (`pf_gdn_front`: grid (conv channels / 256, position
blocks); the ring's three older slots read directly (no seed buffer); a work-group's 256 channels hold two whole
heads, so the l2norm of q and k is done in the work-group with `pf_gdn_l2norm`'s reduction order; the gate's
per-chunk cumsum in a second entry of the same launch's grid tail, or kept as its own launch if the probe shows it
cheaper); `src/runtime/prefill/gdn.cc` (bind under `B70_PF_PHASE2 != old` for Ornith); CMake stanzas at Ornith's and
Qwen3.8's (Flash-Next's) shapes; `tests/prefill/gdn_conv_test.cc` (front == seed + conv + l2norm + gate, every byte).
The qkv||z GEMM writing bf16 instead of fp32 (rounded in its epilogue by the same `rne_bf16`): only if every consumer
(the conv and `pf_gated_head`'s z) rounds the fp32 value once with rne and nothing else reads it - checked in Step 1
by reading both kernels; then it is bitwise too and halves the front's read.

- [x] **Step 1 (Mac):** the kernel, the binding, the test; the qkv||z bf16 check written down in the commit message
  (and done, or not, by its result). `mac_check` green. Commit.
- [x] **Step 2 (box, card 0):** `gdn_conv_test`, `gdn_chunk_test`, the extended G0 compare (Ornith l0 and l0-int8
  bitwise main's), `B70_PREFILL_PROFILE=1` 4k l0.
  - **Gates:** bitwise; the GDN front rows <= 12 ms at 4k l0 (today ~26); Qwen3.8's G0 untouched.
  - **Result (2026-11-12):** bitwise (kernel test and end to end); front 24.8 -> 20.5 ms (the 12 ms bar missed: the
    fp32 qkv||z read is the floor; the bf16 qkv||z was not taken - pf_gemm.cl records it as a wash). Landed.

### Task 3: the GDN intra-chunk kernel (Mac, then box) - JUDGED

Task 1's winning D-arm made production: `pf_gdn_intra` replaces `pf_gdn_A`, `pf_gdn_solve`, `pf_gdn_wu` (and
`pf_gdn_A2_hilo` in the D-a form) - one work-group per (head, chunk), fully parallel over chunks, the scan's inputs
(w, u, A2 hi / lo) written in today's layouts so `pf_gdn_scan_dpas_split` reads them unchanged. Q1-Q4 (the four bf16
roundings FLA's chunked form adds, `pf_gdn_wy.cl`'s header) stay as today unless D2 says otherwise.

**Files:** create `src/kernels/prefill/pf_gdn_intra.cl`; CMake at both shapes; `gdn.cc` (launches 5-8 -> 1 or 2;
`kGdnChunkLaunches`); `tests/prefill/gdn_wy_test.cc` (intra against `gdn_chunk_ref.h` within the Task 1 band: the
structural cases - zero tail chunk entries exactly 0.0f, T's identity case - stay exact); `tests/prefill/gdn_chunk_
test.cc`'s band printed for the new arm.

- [ ] **Step 1 (Mac):** the kernel, the binding, the tests. `mac_check` green. Commit.
- [ ] **Step 2 (box, card 0):** `gdn_wy_test`, `gdn_chunk_test`, `prefill_gate_ornith_{l0,l0-int8}_test` (+ `_c16`),
  `prefill_split_ornith_*`, `ornith_prefill_*` (replay and chunking bitwise: the kernel is deterministic),
  `golden_gate_ornith` (decode continues from the prefill's state).
  - **Gates (judged):** the f32 judge as in Global constraints, on every Ornith prefill gate; `prefill_split_ornith`'s
    near-miss rule; replay bitwise; GDN intra rows <= 28 ms at 4k l0 (today 80.6).
  - FAIL on numerics with Q1-Q4 as bf16: D2.
  - **Result (2026-11-12): STOPPED.** bf16 Q1-Q4 (23.6 ms for 80.6): l0 gates PASS, prefill_gate_ornith_l0-int8 one
    determined row missed. D2 (Q1-Q4 fp32, u / w hi / lo, pf_gdn_scan_hl; 37.4 + 31.1 ms): per-layer far closer to
    fp64, yet l0 cjk and three l0-int8 rows missed. Operator ruling (2026-11-12): the bf16-Q form is the
    default on l0, where every gate passes; l0-int8 keeps the four launches (not supported yet: it flips a determined
    row). D2 (pf_gdn_intra_hl / pf_gdn_scan_hl) removed. Retried on l0-int8 under phase 1's near-miss rule
    (2026-11-12, W4A8 experts): the _c16 gate misses cjk step 15 outside the rule; l0-int8 stays on the old path.

### Task 4: the scan and the gated head (box, then Mac) - per arm

Only what Task 1's per-launch trace and a second look at `pf_gdn_scan_dpas_split` (26.4 ms at 4k: 128 work-groups =
32 heads x 4 column chunks, the only serial kernel) and `pf_gated_head` (12.6 ms) show. Candidates, each its own
commit and its own verdict:
- the scan with 8 column chunks (256 work-groups) - same per-column arithmetic: bitwise;
- the gated head reading z as bf16 (after Task 2's bf16 qkv||z) and o once: bitwise;
- vLLM's fwd_o form (one work-group per head, W S and Q S sharing one load of S, O2 = A2 U recomputed inside) only if
  the trace shows the scan's global traffic, not its dependency chain, is the cost: judged.

- [ ] **Step 1 (box, card 0):** interleaved pairs per candidate at 4k / 16k l0.
- [ ] **Step 2 (Mac, then box):** the winners built, the rest deleted; the gates of the candidate's kind (bitwise: the
  extended G0 compare; judged: Task 3's).
  - **Gate:** a candidate is taken only if it saves >= 3 ms at 4k l0 and passes its numerics gate.
  - **Result (2026-11-12): nothing taken.** The gated head with one sub-group a (position, head) (bitwise end to end:
    prefill_gate_ornith l0 / l0-int8 and ornith_prefill_l0 print main's logs) saved 1.5 ms (13.1 -> 11.6, 3 pairs):
    under the 3 ms bar, removed. The scan with 8 column chunks was not built: at 30 KB of SLM and 256 lanes the
    128 work-groups already fill every Xe-core's 64 thread slots, so 256 would run in two waves of half the work.
    The bf16-z head needs a bf16 qkv||z (recorded as a wash in pf_gemm.cl); the fwd_o form needs a trace showing the
    scan's traffic as the cost, which the profile does not.

### Task 5: residual + RMSNorm in one launch, l0-int8's quantiser folded in (Mac, then box) - BITWISE

Today a norm is `pf_res_fold` (resid += partials and 20 groups' partial sums of squares) then `pf_norm_finish`; 41 ms
over 160 norms at 4k is ~5-8x their bandwidth floor. vLLM's `fused_add_rms_norm` does both in one pass.

**Files:** create `src/kernels/prefill/pf_norm_fused.cl`: one work-group per row (or per 2 rows), the fold, the 20
group sums in today's group order and today's finish arithmetic (so the output is today's, byte for byte); an
`I8_OUT` form that also writes the input norm's rotated int8 rows and scales - `pf_quant_had`'s arithmetic on the bf16
value the norm just rounded (so bitwise the two-launch walk) - and, once phase 1's Task 5 lands, its W4A8 activation
quantiser for the post-attention norm the same way; `src/runtime/prefill/step.cc` (`pf_res_norm` binds it for Ornith
/ Flash-Next under `B70_PF_PHASE2 != old`; Qwen3.8 and the pipeline-parallel hand-off keep the two halves - spec 16c
launches them apart); CMake at hidden 2048 / 2560; `tests/prefill/pf_prep_test.cc` (fused == fold + finish, and
`I8_OUT` == + `pf_quant_had`, every byte).
Before the code: Task 1's trace names what fills l0-int8's 114 ms. If it is a launch timed in the wrong bucket, the
fix is its `profile_wait` and the real cost goes to its own row; the fold then targets whatever that launch is.

- [x] **Step 1 (Mac):** the kernel, the binding, the test. `mac_check` green. Commit.
- [x] **Step 2 (box, card 0):** `pf_prep_test`, `pf_int8_test`, the extended G0 compare.
  - **Gates:** bitwise; the norm row (with the folded quantisers) <= 12 ms at 4k l0 and <= 15 ms at 4k l0-int8.
  - **Result (2026-11-12):** bitwise; norm 41.5 -> 15.8 ms at 4k l0 (12 missed by 3.8; l0-int8's 114 was the setup
    pass mis-bucketed, its real row is the same 41 -> 16). The int8 quantiser fold (I8_OUT) was not built: 5.5 ms at
    stake. Landed.

### Task 6: the route in one pass (Mac, then box) - BITWISE first; JUDGED only by D3

The router (`pf_ab_proj` at N 272, decode's {16, 16} tiling over C / 8 rows a work-group) and decode's `moe_route`
on grid (1, C) cost 36 ms at 4k; a||b uses the same GEMV (10.9 ms). The GEMV's tiling is what makes prefill route
from decode's exact logits (spec 15c), so the bitwise form keeps every logit's reduction order and changes only how
many rows a work-group covers and where the weights come from (SLM-staged router columns, many rows a work-group);
`moe_route`'s arithmetic is fused after it in the same launch (one work-group owns whole rows of all 272 logits).
A DPAS router (vLLM's dense gate GEMM + `topk_softmax`) moves the logits' bits and with them routing near-ties
against decode: only if the bitwise form misses its gate, and only by D3.

**Files:** create `src/kernels/prefill/pf_route.cl`; `moe.cc` (after phase 1's Task 3), `step.cc` (a||b); CMake;
`tests/kernels/pf_moe_test.cc` (router rows bitwise decode's `gemv_bf16`, route rows exactly decode's `moe_route` of
them), `tests/prefill/pf_gemv_test.cc`.

- [ ] **Step 1 (Mac):** the kernel, the bindings, the tests. Commit.
- [ ] **Step 2 (box, card 0):** `pf_moe_test`, `pf_gemv_test`, `ornith_prefill_*` (prefill-vs-decode routes), the
  extended G0 compare.
  - **Gates:** bitwise; route + a||b <= 12 ms at 4k (today ~47).
  - **Result (2026-11-12):** bitwise tiling (pf_route.cl, 16 rows x 2 column tiles a work-group) bound for the router
    and a||b: 0.69x / 0.65x, end to end bitwise; ~47 -> ~33 ms, the 12 ms bar missed. D3 (DPAS router) built and
    tried: no faster as built and it missed the f32 judge on l0-int8 - not taken, removed.

### Task 7: the dense linears (Mac, then box) - only if Task 1's dense arm passed its gate - JUDGED

The winning dense arm made production for Ornith's (and Flash-Next's) linear shapes as new binaries beside
`pf_gemm` / `pf_gemm_i8` (which Qwen3.8 and the parked models keep). Reduction order changes, so judged.

- [ ] **Step 1 (Mac):** kernel, bindings, `pf_gemm_test` / `linear_i8_test` arms. Commit.
- [ ] **Step 2 (box, card 0):** Task 3's judged gates; dense row >= 10 % below Task 6's at 4k on the backend it targets.
- **Result (2026-11-12): not run.** Task 1's dense arm was not built (the operator asked for no probes beyond a gate's
  needs). The slab walk's GEMMs run at 75 % of the bf16 peak on l0 and 57 % of the int8 peak on l0-int8 (the profile),
  so a >= 10 % arm is unlikely on l0. The int8 side's headroom (57 %) is the one place a probe could still pay.

### Task 8: Flash-Next on the phase 2 kernels (Mac, then box)

Flash-Next's 36 GDN layers bind Tasks 2-4's binaries at Qwen3.8's shape (`gdn_chunk_q4`), its norms Task 5's at hidden
2560 where its hyper-connection walk uses `pf_res_fold` / `pf_norm_finish` (its `q4_hc_combine_norm` is its own kernel
and stays), its route Task 6's at 512 experts top-10.

**Files:** `src/runtime/qwen4exp/qwen4exp_prefill.cc`, `qwen4exp_plan_test` (launch counts),
`tests/kernels/qwen4exp_pf_kernels_test.cc` (the new arms at the real shapes, synthetic weights).

- [ ] **Step 1 (Mac):** the bindings, the tests. Commit.
- [ ] **Step 2 (box, card 0):** `r33.k1`, `r33.prefill` (both synthetic forms; two prefills and chunks of 64 bitwise
  each other), `r33.golden` (the tie-aware token gate), `r33.golden_real` at `--layers 18` (Intel's checkpoint),
  `r33.speed`'s pp4096 at `--layers 18` interleaved against main.
  - **Gates:** bitwise tasks bitwise main's Flash-Next output; Task 3's GDN judged by r33.golden's gate and
    r33.golden_real; pp4096 at `--layers 18` faster than main's (the phase table printed and recorded).
  - **Result (2026-11-12):** Flash-Next binds the GDN front (bitwise, on main since Task 2) and the router (K 2560,
    N 528) and a||b on pf_route.cl's tiling (bitwise). k1 / prefill / golden synthetic and qwen4exp_prefill_intel_test
    (routes against the recorded engine trace) 19 / 19 PASS, the synthetic logs identical to main's. pp4096 at
    --layers 18 (card 1, 2 interleaved pairs): 4,663 -> 4,698 t/s (+0.7 %, within noise). The intra-chunk kernel is
    not bound for Flash-Next: it passed the synthetic and golden gates (golden_real 77 vs main's 74 determined rows
    over three prompts, a test main fails too), but qwen4exp_prefill_intel_test matches routes bitwise against the
    engine trace recorded in oracle-out-q4exp-f32-aligned-L18. That reference has to be re-made on the CPU first.
    The one-sub-group gated head (Task 4) saved under 3 ms on Ornith, so it was not bound here either.

### Task 9: the record, the clean-up, the final gate (box idle, then Mac)

- [ ] **Step 1 (box, card 0, idle):** interleaved pairs, main's build (control) against the branch head (candidate),
  median of 3, `b70-decode --bench --tg 32`: Ornith pp4096 and pp16384 on `l0` and `l0-int8`; then the branch alone
  with `B70_PREFILL_PROFILE=1` for the phase table against the budget above.
  - **Gate (final):** pp4096 **>= 11,000 t/s on l0-int8** and **>= 9,500 t/s on l0**; the 16k numbers recorded. A miss
    keeps the item open with the measured phase table against the budget (which phase is over, by how much).
- [ ] **Step 2 (box):** **decode unchanged:** no decode binary moved (`g0.sha`), `ornith_decode_test` PASS (replay
  bitwise, R2), plain decode at depth 4096 within 1 % of 142.8 t/s (interleaved). **Other models:** G0 (row 0) PASS
  against `--baseline main` - Qwen3.8 bitwise (unless D1 switched it, then its own gates: `prefill_gate_l0_test`,
  `prefill_gate_int8_test`, `prefill_split_*` judged, its pp4096 not slower), every K2 / Kolibri / Bolt binary
  identical; the Ornith rows (r13.gates, r13.split, r13.prefill, r10.gates) and Flash-Next's r33 rows PASS.
- [ ] **Step 3 (Mac):** delete the losing entries and `B70_PF_PHASE2` (Ornith-only binaries go; the Qwen3.8-shape
  `pf_gdn_*` binaries stay while Qwen3.8 binds them); BENCHMARKS "Ornith 1.5 MoE (spec 15)" and Flash-Next rows, the
  profile doc's "after phase 2" table, docs/12-kernels.md's GDN section, open-work -> Closed recently. Commit; offer
  the ff-merge to main.

- [ ] **Step 4 (box, idle, both cards; operator 2026-11-12):** once plan 1 and this plan are both on main, a full
  validation run on main, unattended: `tools/box_validate.sh` with the opt-in stages of open-work's "28 speculative
  serving" row added (r2.auto_rows, r12.a4, r6.greedy_long, r8.auto_rows, r16.mtp_rows, r27.cli_mtp, r55.serve,
  r55.decode, r56.pp_serve, r70.d2, r70.speed - the exact names and the `--with` syntax from `--list`). Report PASS /
  FAIL / SKIP per row. The f32 judge decides each first difference; one with no reference yet SKIPs with its context
  dumped, and `tools/box_validate/f32_refs.sh <data> forced-qwen` makes the reference on the CPU afterwards. This
  closes open-work item 15 and the speculative-serving row together.

**Which tasks need the box:**

| task | Mac | box | numerics |
|---|---|---|---|
| 1 trace + probes | step 1 | step 2 (card 0) | error vs fp64 per arm |
| 2 GDN front | step 1 | step 2 | bitwise |
| 3 GDN intra | step 1 | step 2 | judged |
| 4 scan, head | step 2 | steps 1-2 | per arm |
| 5 norms | step 1 | step 2 | bitwise |
| 6 route | step 1 | step 2 | bitwise (judged only by D3) |
| 7 dense (conditional) | step 1 | step 2 | judged |
| 8 Flash-Next | step 1 | step 2 | as the tasks it binds |
| 9 record | step 3 | steps 1-2 (idle) | - |

Test and benchmark runs (Task 1 step 2, the box steps' measurements) go to a Sonnet agent; implementation to Opus.

## Operator decisions

- **D1 (after Task 3 passes on Ornith):** switch Qwen3.8's GDN prefill to the new kernels too? Its output changes
  (judged by its gates, G0's baseline moves) for an expected ~-60 ms a 4k prefill at its 48 v-heads (derived).
  Recommendation: yes, as its own small task after this plan, not inside it.
- **D2 (only if Task 3 fails its numerics with Q1-Q4 as bf16):** carry Q1-Q4 in fp32 (accuracy over FLA parity; the
  2026-10-31 ruling leaned that way, the round-2 trial was parked on branch `g0-fix-gdn-fp32` at -1..2 % prefill).
- **D3 (only if Task 6's bitwise route misses its gate):** a DPAS router whose logits differ from decode's in the last
  bits, judged by the f32 judge and the prefill-vs-decode route diagnostic.
- **If Task 9 misses l0's 9,500:** the measured table says whether the miss is phase 1's MoE GEMM (budget 145 ms) or
  this plan's phases; the next step is the operator's call.
