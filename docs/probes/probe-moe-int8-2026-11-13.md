# Probe: is form A (bf16 mixer linears + W4A8 experts + one-launch delta-net kernel + GEMM router and a||b) clean and fast on Ornith - 2026-11-13

Plan: `docs/superpowers/plans/2026-11-13-moe-int8-prefill.md`, Task 1. Branch `moe-int8-probes` (probe branch, not merged; base =
phase 3 + main). One card (card 1), `tools/gpu_lock.sh card 1`.

Arms (env switches on this branch only): `l0`; `l0-int8` today; **A0** = the four mixer linears bf16
(`B70_PF_I8_MIXER_IN=dense`); **A1** = A0 + the one-launch delta-net kernel (`B70_PF_I8_GDN=intra`); **A2** = A1 + phase 3's
GEMM router and a||b (`B70_PF_I8_ROUTE=gemm`) = form A.

## Gates (pre-registered, copied from Task 1 Step 3 before any run)

- **Split (2,600 ids vs f32), A0 / A1 / A2:** logits and all-rows median <= 2x the bf16 reference's (the test's own rule,
  `kCosSlack`); median <= 1.15x; no printed delta-net state above 1.3x `l0`'s own multiple for that layer.
- **Short gates, A0 / A1 / A2:** `prefill_gate_ornith_l0-int8_test` and `_c16` pass with no more rule-passed rows than
  today's (one each); every such row printed with its f32 margin and both cosines.
- **Serving, A2 (and A0 if A2 fails):** 0 FAIL of 39. Control, at no extra cost: at every first difference the judge
  already has the f32 reference of `l0`'s own continuation. For each request, report whether `l0`'s continuation from that
  point passes the same rule. A request where `l0`'s side is rejected too is reported as "both sides rejected" and counted
  as the floor, not as form A's FAIL.
- **Speed, A2 vs `l0`, pp4096 interleaved:** >= 1.08x; with `B70_PREFILL_PROFILE=1` the phase table.
- Reported, not gated: which of today's six failing requests pass on A0 (the open cause question of item 16), the
  tool-call test's identical / invalid counts against bf16 (today 20 / 36, 1 invalid).

Stop rules (plan): A2 serving with any FAIL that `l0`'s own control does not have -> D2; the one-launch kernel failing a
gate on form A while the four launches pass -> D3; speed below 1.05x `l0` -> record, "no speed case".

Stages: `r86.split`, `r86.gates`, `r86.serve`, `r86.speed` (`tools/box_validate/stages.sh`, opt-in, `--device 1`).

## Results (WIP, 2026-11-12 evening; box busy: card 0 and CPU reference jobs ran beside every arm)

Logs on the box: `~/b70/logs/moe-int8/` (split_*, gates_*, serve_*, speed.*). Driver scripts: `.pmp/` in the box tree (untracked).

### Split, 2,600 ids vs f32 (`prefill_split_ornith_*`, B70_SPLIT_VERBOSE=1), x = engine / bf16 reference's distance to f32

| arm | logits x | all-rows median x | worst delta-net state x (layer) | states > 1.3x `l0`'s own multiple | worst FA row x |
|---|---:|---:|---|---|---:|
| `l0` | 0.57 | 1.01 | 2.12 (4) | - | 1.64 |
| `l0-int8` today | 0.59 | 1.39 | 6.38 (0) | 18 of 30 layers (up to 6.33x at layer 1) | 2.35 |
| A0 | 0.66 | 1.06 | 2.02 (4) | layer 18: 0.66 vs `l0`'s 0.47 = 1.40x | 1.65 |
| A1 | 0.81 | 1.05 | 2.24 | none (max 1.24x) | 2.07 |
| A2 | 1.04 | 1.04 | 2.19 | layers 20 (1.15 vs 0.87 = 1.32x) and 29 (1.22 vs 0.92 = 1.33x) | 1.64 |

Verdict against the pre-registered gate: logits <= 2x and median <= 1.15x pass for A0 / A1 / A2 (A2 logits 1.04x is the
highest, it was 0.57x on `l0`). The delta-net state sub-gate (no state above 1.3x `l0`'s own multiple) is **missed
narrowly by A0 (layer 18) and A2 (layers 20, 29)**, by the letter; every one of these multiples is itself at or near 1x of
the bf16 reference's distance (0.66 against 0.47; 1.15 against 0.87; 1.22 against 0.92), where the ratio of two small
numbers moves most. A1 passes. `l0-int8` today fails it widely (the drift diagnosed in the plan, section 1).

### Short gates (`prefill_gate_ornith_l0-int8_test` c0 / `_c16`, the test's own pfi8 judge)

| arm | c0 | c16 | near-tie rule-passed rows (both tests) | determined MISMATCH rows |
|---|---|---|---:|---|
| `l0-int8` today | pass | pass | 5 (2 + 3 lines of the log; 1 + 1 listed are the gate's accepted) | 0 |
| A0 | **FAIL** | pass | 4 | row 15: golden 99737, engine 97663 = f32's 97663 (f32 cos 0.998551, engine's own cos to bf16 0.998997) |
| A1 | pass | pass | 5 | 0 |
| A2 | **FAIL** | **FAIL** | 7 | c0 rows 19 (golden 855, engine 13 = f32's, cos 0.995988) and 15 (as A0); c16 row 15 |

Caution on this gate: its judge (`pfi8`) simulates rotated int8 on the mixer linears, so on arms with bf16 mixer linears
the "MISMATCH" rows are rows where the engine agrees with the plain f32 reference and the simulator does not. The plain-f32
judge run is below.

### Serving (b70-serve, `--spec off` as the 2026-11-12 run (serve_run.sh's default), 39 requests, 256 ids, `--max-len 16384`)

Run 19:39-19:46: all four arms answered all 39 requests. Against `l0` (this build): `l0-int8` today 1 identical; A0 0
identical; A2 3 identical. The f32 references of the first differences did not exist for A0 / A2 (the CPU reference
queue: one 15.6 min context at a time, 220 contexts for A0 + A2 + today = 56 h; started for A2's and A0's six
previously-failing requests first, then stopped for the operator's quiet-box run). Judge result so far: A0 0 judged,
A2 0 judged, `l0-int8` today 1 FAIL decided without a reference wait (`toolcall/t3_rename-loader`, 2 speculative / 0 plain
ids rejected, worst miss 9.69 ulp: `l0`'s side passes, so it is not a "both sides rejected" case).

### Speed, pp4096, `b70-decode --bench` (t/s), interleaved, BUSY box (load 36-49, card 0 running another tool's job, a CPU reference job)

The run was cut short in round 2 (stopped by an external kill at 20:04) after warm-ups and round 1 of all arms:

| arm | warm-up | round 1 | round 2 |
|---|---:|---:|---:|
| `l0` | 9,251 | 9,253 | - |
| `l0-int8` today | 8,798 | 8,815 | - |
| A0 | 8,591 | 8,518 | - |
| A1 | 9,744 | 9,722 | - |
| A2 | 9,731 | 10,142 | 10,485 |

A2 / `l0` = 1.10 (round 1), 1.13 (round 2): above the 1.08x gate and the 9,600 t/s target, on one to two samples under a
busy box; the median-of-3 and the phase tables are to be redone on a quiet box.

### Plain-f32 judge re-run of the short gates (no accept list; 2026-11-12 night, card 0, build with phase 3 head 98f225cb merged)

Every arm and `l0` through `prefill_gate_test --judge=oracle-out-ornith-f32 --accept=` (c0 / c16). MISMATCH rows (engine id,
bf16 golden id, f32 top): `l0` c16 row 15 (golden 99737, engine = f32 = 97663); A0 c0 row 15 (same); A2 c0 rows 19 (golden
855, engine = f32 = 13) and 15 (same as above); A2 c16 interrupted. Today's `l0-int8` (c0, c16), A1 (c0, c16) and A0 c16:
no MISMATCH. So the A0 / A2 FAILs of the short gates are **not the simulated-int8 judge's artefact alone**: the same
row 15 fails `l0` itself at c16 on the plain judge, and in every flagged row the engine picks the f32 top while the bf16
golden is the odd id out. They are rows where the bf16 golden and f32 disagree (the accept list normally holds them), not
tokens that are wrong against f32. Row 19 (A2 c0) is new to A2 (A2's own router / a||b rounding), same kind.

### Status at the park (operator, 2026-11-12 night)

Not measured: the serving verdicts for A0 / A2 (no f32 references were built: the reference jobs were stopped twice for the
operator's quiet-box runs and then parked; 110 references exist, none for A0 / A2's contexts), the "both sides rejected"
floor, the six old FAILs on A0, and the median-of-3 speed with the A2 phase table (the speed run of 20:03 was cut in round 2;
the rerun was queued behind the plain gates and parked). See HANDOFF.md.
