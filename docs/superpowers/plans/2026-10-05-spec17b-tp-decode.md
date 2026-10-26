# Spec 17b - tensor parallel: Bolt decode across two cards

**Conditions (operator, 2026-10-23; for Bolt - superseded for Qwen3.8 by the 2026-10-27 status below):** this plan starts only when **spec 17 §9 records GO** - P0 (plan 17a) ran after
`r22.devices` / `r22.p1`, **Bolt's rows 59-62 passed**, and P0 measured one Bolt token's crossings at **<= ~1 ms**
(spec 17 §3) - and the operator says go (one implementing plan at a time).

**Status (2026-10-27): PARKED -> next after spec 28 (28a / 28b), spec 19's plan 19f and spec 17a's GO.** Retargeted
to Qwen3.8 first (spec 17 §R2): this plan is rewritten for the main engine - `b70-decode <qwen3.8> --tp 2`, then Agnes:
spec 17 §R2.5's split re-fused per card at load, the all-reduce on the route 17a chooses, the vocabulary-split head, the
verify lists on spec 28's `gemm_sm` at per-card shapes - when 17a records GO and the operator says go. Box queue row
**72** (the speed track's rows are 68-75, spec 28's header). The 2026-10-23 Bolt-first text below is kept for Bolt's later
TP plan (Bolt has no drafter since 2026-10-27).

**Status (2026-10-23, superseded 2026-10-27): PARKED - begins on the conditions above.** Rewritten 2026-10-23 for Bolt first (spec 17 §R);
the 2026-10-05 version (Qwen3.8 in the main engine) is in git history. The dense models are not in this plan.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** `b70-decode <bolt> --tp 2`: each card holds its slice of every block (spec 17 §4.2) and runs its own captured
list at per-card shapes; the crossings use P0's chosen mechanism and direction in a fixed fold order so both cards
hold a bitwise-identical residual; the MoE by P0's scheme (spec 17 §4.3); the head split by vocabulary with the argmax
exchange; T1, T2, T4; D1-Bolt, D3.

**Architecture:** spec 17 §4.1-§4.6, §4.9, §4.10. `model::BoltDesc` gains a `tp_rank` / `tp_size` view deriving every
per-card shape; `loader::load_bolt` gains a TP placement (both devices hold every block, each its re-fused slice -
`in_proj`'s five column ranges, `q‖k‖v`, `fc1‖shared_up` with the replicated fc1 part - and the replicated router,
`fc1` / `fc2`, embedding, `norm_f`); `runtime::bolt::tp_plan` beside `runtime::bolt::plan` (plan == allocation);
`BoltEngine` with one list per card, the crossing landing buffers (page-aligned, their own allocations), the fold
kernel's TP form (`nb_res_norm` remote sources or the fold-then-send epilogue, per P0), the routed part of the MoE
crossing.

**Tech Stack:** C++17, OpenCL C, Level Zero, CMake / ctest.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§4.1-§4.6, §4.9, §4.10, §5 T1 / T2 / T4,
§6 D1-Bolt / D3, §9 the verdict). Precedents: spec 16b (one context over both cards, `pp_send` / `pp_recv`, bounded
fences), spec 25b / 25c (the loader, `BoltEngine`, the `nb_*` family, B3's bars).

## Dependencies and branch points

- Main with 17a's verdict (GO) merged. Branch **`spec17b-tp-decode`** from main.
- Box: rows 59-62 PASSed; `bolt_g128` and `oracle_bolt` present; the N 12 synthetics (`bolt_synth`).

## Global Constraints

- Signed commits after every step; no merge, no push. Box tree automatic; `tools/box.env` copied if missing, never
  committed; `oracle-out*` symlinked. No `rm -rf`.
- **`--tp 1` is today's Bolt engine bit for bit:** every existing kernel binary unchanged (`tools/kernel_cmdlines`:
  additions only; G0's sha). Per-card shape variants are new binaries.
- `--tp 2` with `--pp 2`, `--replica`, `--pipeline-split` or `--pipeline-handoff` refused by name before any device.
- **Every two-card stage is `gpu`** (`tools/gpu_lock.sh box`); host tests `cpu`. Timed arms: idle box, interleaved pairs
  after a warm-up, median of 3; device 1's gap recorded.
- Shapes only from `BoltDesc`; no Bolt constant in a kernel or a planner.

## Review Focus

1. **Slicing on boundaries:** a host test slices synthetic packed tensors and checks each card's re-fused slice
   dequantises to exactly the rows / columns spec 17 §4.2 names (heads 0-63 / 64-127; B / C groups 0-3 / 4-7; kv head
   0 / 1; shared-expert halves; every K slice a whole number of g64 groups; EP's expert halves or intra-expert's I / 2).
2. **T1:** after every block the two cards' residuals and MoE route sets byte-identical (a debug tap compares them),
   golden prompts + 64 tokens; a route mismatch fails.
3. **T2:** against `--pp 2` (bitwise one card's order) under the tie-aware gate and B3's bars (routes with near-ties,
   the SSM state / h cosines), short / 4k / agentic.
4. **The argmax exchange:** the same id on both cards, ties to the lower id.
5. **T4:** a missing card, no peer access, a crossing timeout, a stale stamp: clear errors within the bound; the
   engine marked failed until `reset()`.

---

### Task 1: the TP view, the slices and the plan (Mac)

**Files:** `src/model/bolt.{h,cc}` (the TP view), `src/loader/bolt_loader.{h,cc}`, `src/loader/bolt_repack.{h,cc}`,
`src/loader/bolt_layout.h`, `src/runtime/bolt/bolt_sizes.{h,cc}`, new `tests/loader/bolt_tp_slice_test.cc`,
`tests/runtime/bolt_tp_plan_test.cc`.

- [ ] **Step 1: failing tests** (Review Focus 1; the plan's bytes per card - spec 17 §4.6's ~22.5 GB derived - and
  `--max-len auto` from the tighter card).
- [ ] **Step 2: implement, PASS.** Commit `git commit -S -m "loader: Bolt sliced per card at load, the TP plan (spec 17b)"`.

### Task 2: the crossing and the per-card lists (Mac)

**Files:** `src/kernels/bolt/nb_prep.cl` (the fold's TP form), `src/kernels/bolt/nb_moe.cl` (the routed part of the
crossing), `src/kernels/CMakeLists.txt` (per-card variants), `src/runtime/bolt/bolt_capture.{h,cc}`,
`src/runtime/bolt/bolt_engine.{h,cc}`, `src/runtime/bolt/bolt_buffers.{h,cc}`, new `tests/runtime/bolt_tp_protocol_test.cc`
(two host threads), `tests/kernels/bolt_tp_kernels_test.cc` (the twin: the TP fold = the one-card fold's sum in the
fixed order).

- [ ] **Step 1: failing tests**, then implement; `tools/mac_check.sh --base main --kernels` (the new variants on the
  Mac's OpenCL, INDICATIVE). Commit `git commit -S -m "runtime: Bolt decode across two cards - the crossing, the per-card lists, the head exchange (spec 17b)"`.

### Task 3: the CLI and the refusals (Mac)

- [ ] `b70-decode <bolt> --tp 1|2` (`--tensor-parallel-size`), the exclusivity refusals, `cli_reject_tp_*` tests.
  Commit `git commit -S -m "cli: b70-decode <bolt> --tp 2 and its refusals (spec 17b)"`.

### Task 4: the box queue row (proposed 69, or the next free row)

- [ ] Row and stages (`test_box_validate.py` passes):

```text
row 69 "spec 17b - Bolt decode under tensor parallel (plan 17b)"
stage r69.k0      69 default cpu  -                    g0.sha,g0.bitwise,g0.suite "D3: --tp 1 is today's engine - every existing binary identical"
stage r69.host    69 default cpu  -                    -          "bolt_tp_slice_test, bolt_tp_plan_test, bolt_tp_protocol_test, cli_reject_tp_*"
stage r69.k1      69 default gpu  -                    r69.k0     "bolt_tp_kernels_test on both cards: the TP fold = the fixed-order sum, the routed crossing both forms"
stage r69.synth   69 default gpu  bolt_synth           r69.k1     "bolt_tp_decode_test on the N 12 synthetics: plan == allocation per card, T1 per block, T2 vs --pp 2"
stage r69.p4      69 default gpu  bolt_synth           r69.synth  "T4: missing card, no peer access, a dropped crossing, a stale stamp - bounded, named"
stage r69.golden  69 default gpu  bolt_g128,oracle_bolt r69.synth "SergiioB's g128 under --tp 2: T1, T2 (B3 short / 4k / agentic) against --pp 2"
stage r69.speed   69 optin   gpu  bolt_g128            r69.golden "D1-Bolt: --tp 2 vs --pp 2 at depth 886 / 8192 / 32000 / 65536, idle, interleaved, median of 3; bar >= 1.4x at 8192 (proposed)"
rownote 69 "After spec 17 §9's GO. Every two-card stage is gpu."
```
- [ ] Commit `git commit -S -m "docs: box queue row 69 - Bolt TP decode (spec 17b)"`.

**Gate for the plan:** host tests green; on the box T1 bitwise, T2 green against `--pp 2`, T4 bounded, D3 (G0), D1-Bolt
measured and recorded in docs/BENCHMARKS.md "Tensor parallel (spec 17)".
