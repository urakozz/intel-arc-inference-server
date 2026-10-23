# Spec 17a - tensor parallel P0: the links, the peer rate, the crossing latency at each target's count, the MoE balance, the go / no-go line

**Conditions (operator, 2026-10-23):** this plan **starts only after `r22.devices` and `r22.p1` have PASSed on the box**
(spec 16's first two-card rows: two cards in one process, peer access, the `copy` / `peer` hand-offs bitwise). It is
standalone: it depends on no spec 16a artefact (16a never ran; `docs/probe-multi-gpu-2026-10-05.md` and
`tools/probe/probe_p2p.*` do not exist) and owns its probe.

**Status (2026-10-23): PARKED - begins on the conditions above.** Rewritten 2026-10-23 for the Bolt-first ruling
(spec 17 §R); the 2026-10-05 version (Qwen3.8 shapes, "building on 16a's `probe_p2p`") is in git history.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the numbers spec 17 §3's rule needs, and the rule applied: each card's PCIe link, the peer copy rate both
ways (push and pull), **the per-crossing latency** of three crossing mechanisms inside replayed lists at each target's
count and size (Bolt first), the per-card Bolt step at TP shapes, and the MoE balance from 25a's route records. The
plan ends with a GO / NO-GO line in spec 17 §9.

**Architecture:** spec 17 §1, §3, §4.1, §4.3. A new probe `tools/probe/probe_tp` (C++ / Level Zero, one process, one
context over both cards - spec 16b's `l0::Context` device view) with its own kernels in `tools/probe/probe_tp.cl`
(the crossing arms; spec 16 §2's two P2P rules: a page-aligned landing allocation of its own, system-scope atomics,
counters in device memory, system-scope loads of the landed data); a host twin of the crossing protocol on two
threads; `tools/spec/tp_expert_balance.py` over 25a's route records; 25c's `probe_bolt_step` gains `--tp-shapes`
(each linear at its per-card shape, spec 17 §4.2).

**Tech Stack:** C++17, OpenCL C, Level Zero; Python 3 (the oracle image) for the balance.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§1, §3, §4.1-§4.3, §9). Precedents:
plan 22a (`offload_links.sh`, `probe_offload`'s interleaving and bounded waits), plan 25c's P0 (`probe_bolt_step`,
`bolt_step_shapes.h`), plan 26a (`expert_union.py`).

## Dependencies and branch points

- Main with spec 16b-16d (merged), 25a-25e (merged), 26a (merged). Branch **`spec17a-tp-probe`** from main.
- **Box:** `r22.devices` and `r22.p1` PASSed (the conditions). Row 57's `r57.golden` for Bolt's route records
  (`oracle-out-bolt`, data key `oracle_bolt`); without it Task 4 runs on the N 12 synthetics' records only and says so.
- Kolibri's route records need spec 20b's real checkpoint (decision 1 pending): Task 4 reports Kolibri as SKIP until
  then.

## Global Constraints

- Signed commits after every step; no merge, no push. Box tree automatic; `tools/box.env` copied if missing, never
  committed. No `rm -rf`.
- **Every TP stage is two-card, so `gpu`** (the whole box through `tools/gpu_lock.sh box`); the host / CPU stages are
  `cpu`. Detached and polled (`tools/probe/detach.sh`). **Idle box** for every timed arm (both cards free of other DRM
  holders, `uptime` and the idle grade recorded); interleaved pairs after a warm-up, **median of 5** with the range.
- No engine code changes: `kernel_cmdlines` +0 / -0 / ~0 for `src/` (the probe's kernels are `tools/probe`'s; G0's sha
  is the check). `probe_bolt_step --tp-shapes` adds stand-in shapes only to `tools/probe/bolt_step_shapes.h`.
- Every number written with its grade (spec 17's legend); the operator-stated PCIe 3.0 x16 is replaced by the measured
  line once `r68.links` runs.

## Review Focus

1. **Latency is the number** (spec 17 §4.1): at 8-16 KiB a crossing the transfer is ~1 µs (derived); the arms must time
   the whole crossing - the GEMV's last store, the peer write landing, the release, the peer noticing the flag, the fold
   - inside two replayed lists, minus the same lists without crossings.
2. **Push vs pull:** the peer write (push) and the peer read after a flag (pull) both measured; the root complex may
   make pulls much slower. The mechanism the verdict picks names which.
3. **Bitwise on both cards in every arm:** the folded rows identical on card 0 and card 1, over 10^4 replays; a stamp
   per crossing (sequence number) so a stale landing buffer is detected, never consumed - the vLLM zero-byte lesson
   (spec 16 §2).
4. **Sizes from the descriptors**, never constants: Bolt from `model::BoltDesc` (hidden 4096, S per linear - S1
   PROVISIONAL until `r60.p0`'s sweep; the MoE crossing's routed part both forms, spec 17 §4.3), Kolibri from
   `Kolibri1Desc`, Qwen3.8 from `ModelDesc` (S4: 80 KiB a row under remote partials).
5. **The balance uses real routes:** per (row, MoE layer) sets from the records, the busier card's bytes from each
   layer's own block size (E / I differ by layer), M = 1..8 consecutive rows for the verify.

---

### Task 1: the probe and its host twin (Mac)

**Files:**
- Create: `tools/probe/probe_tp.cc`, `tools/probe/probe_tp.cl`, `tools/probe/tp_shapes.h` (the targets' crossing
  counts and sizes from the descriptors), `tests/kernels/tp_shapes_test.cc` (beside `probe_bolt_step_shapes_test`),
  `tests/runtime/tp_crossing_protocol_test.cc` (beside `pp_protocol_test`: the push and pull protocols on two host
  threads - stamps, bounded spins, a dropped write detected)
- Modify: `tools/probe/CMakeLists.txt`, `tests/CMakeLists.txt`, `tools/probe/bolt_step_shapes.h` + `probe_bolt_step.cc`
  (`--tp-shapes`)

**Interfaces:** `probe_tp links|peer|cross|head|prefill [--model bolt|kolibri|qwen38] [--mech remote|fold|allreduce]
[--dir push|pull] [--m 1|4|8] [--rounds N] [--replays N]`; every wait bounded (a timeout prints the arm and exits 1).

- [ ] **Step 1: failing tests** - `tp_shapes_test` (Bolt 88 + 1 crossings, 16 KiB at S1, the MoE crossing's 4 / 36 KiB
  routed forms; Kolibri 100 + 1; Qwen3.8 128 + 1 at 80 / 20 KiB; all from the descriptors), `tp_crossing_protocol_test`.
- [ ] **Step 2: implement, PASS;** `tools/mac_check.sh --base main --quick` green. Commit
  `git commit -S -m "probe: probe_tp - the TP crossing arms (push / pull, remote partials / fold-then-send / all-reduce) and their host twin (spec 17a)"`.

### Task 2: links and the peer rate (box, `r68.links`, `r68.peer`)

- [ ] **Step 1:** `offload_links.sh` idle and `--under-load` (both cards' LnkCap / LnkSta, every bridge to the root port);
  the line replaces spec 17 §1's operator-stated row. A STOP line (below Gen3 x8) is a NO-GO input.
- [ ] **Step 2:** `probe_tp peer`: copy engine 0 -> 1, 1 -> 0, both at once; kernel push and pull; 4 KiB to 64 MiB;
  immediate and replayed. Record in `docs/probe-tp-<date>.md`. Commit
  `git commit -S -m "probe: the two cards' links and peer rates, push and pull (spec 17a)"`.

### Task 3: the crossing latency (box, `r68.cross`, `r68.head`)

- [ ] **Step 1:** `probe_tp cross` for each mechanism x direction, at Bolt's 89 crossings and sizes (M = 1, 4, 8), then
  Kolibri's and Qwen3.8's counts: per-crossing latency and per-token total, bitwise on both cards over 10^4 replays
  (Review Focus 1-4). `probe_tp head`: the vocabulary-split argmax exchange (ties to the lower id, the same id on both).
- [ ] **Step 2:** `probe_bolt_step --tp-shapes --depth 4096 --pair` on one card: the per-card step at TP shapes (EP and
  intra-expert MoE stand-ins at half width). Commit
  `git commit -S -m "probe: the TP crossing latency at Bolt / Kolibri / Qwen3.8 counts and the per-card Bolt step (spec 17a)"`.

### Task 4: the MoE balance (box CPU, `r68.balance`)

**Files:** Create `tools/spec/tp_expert_balance.py` (+ `test_tp_expert_balance.py`, a synthetic route set with known
answers).

- [ ] **Step 1:** from `oracle-out-bolt`'s route records: per placement (even / odd, contiguous halves, frequency-balanced
  on a held-out half) the busier card's routed bytes per (row, layer) - mean, p50, p90 per E / I bucket - at M = 1 and
  the per-card expert union at M = 2..8; intra-expert TP's even half beside it (spec 17 §4.3). Commit
  `git commit -S -m "spec: TP expert balance from Bolt's routes (spec 17a)"`.

### Task 5: the verdict

- [ ] **Step 1:** spec 17 §9 "P0 verdict": the tables, the derived TP decode per target (per-card step + crossings),
  the mechanism and direction chosen, the MoE scheme proposed per spec 17 §4.3's rule, the prefill crossing (opt-in
  `r68.prefill`) recorded for 17c, and **the line**, by spec 17 §3's rule:
  - **GO** when (a) Bolt's rows 59-62 have passed, (b) the chosen crossing is bitwise on both cards over 10^4 replays
    with stale writes detected, and (c) one Bolt token's 89 crossings (M = 1, inside replayed lists, idle box) cost
    **<= 1.0 ms**, measured;
  - **the operator's call** when (c) is between 1.0 and 1.2 ms;
  - **NO-GO** (TP stays parked; the record names the failed condition and the margin) when (b) fails, (c) exceeds
    1.2 ms, or a link trains below Gen3 x8.
  - Kolibri: the same rule at 101 crossings, stated as pending until spec 20b's checkpoint and Bolt's record.
  `docs/README.md`: the probe record's line. Commit `git commit -S -m "spec 17: TP P0 verdict - <GO|NO-GO|operator's call>"`.

### Task 6: the box queue row (when built)

- [ ] **Step 1:** queue row **68** (or the next free row then) in `box-validation-queue.md` and its block in
  `tools/box_validate/stages.sh` (`python3 tools/box_validate/test_box_validate.py` passes):

```text
row 68 "spec 17a - the TP P0: links, peer rates push / pull, the crossing latency at Bolt / Kolibri / Qwen3.8 counts, the per-card Bolt step, the MoE balance, the go / no-go line (plan 17a)"
stage r68.k0      68 default cpu  -                        g0.sha,g0.bitwise,g0.suite "G0: no src/ binary changes (the probe's kernels are tools/probe's)"
stage r68.host    68 default cpu  oracle_image             -                     "tp_shapes_test, tp_crossing_protocol_test, test_tp_expert_balance.py"
stage r68.links   68 default gpu  -                        r22.devices           "offload_links.sh idle and --under-load: both cards' links and bridges; a STOP line stops the row"
stage r68.peer    68 default gpu  -                        r22.p1,r68.links      "probe_tp peer: copy engine both ways and both at once, kernel push and pull, 4 KiB - 64 MiB, immediate and replayed"
stage r68.cross   68 default gpu  -                        r68.peer              "probe_tp cross: three mechanisms x push / pull at Bolt's 89 crossings (M 1 / 4 / 8), Kolibri's 101, Qwen3.8's 129; bitwise over 10^4 replays; idle, median of 5"
stage r68.head    68 default gpu  -                        r68.peer              "probe_tp head: the vocabulary-split argmax exchange, ties to the lower id"
stage r68.step    68 default gpu  -                        r68.k0                "probe_bolt_step --tp-shapes --depth 4096 --pair: the per-card Bolt step at TP shapes"
stage r68.balance 68 default cpu  oracle_bolt,oracle_image -                     "tp_expert_balance.py over 25a's route records: EP placements vs intra-expert, M 1..8"
stage r68.prefill 68 optin   gpu  -                        r68.peer              "probe_tp prefill: a 2048-row crossing (fp32 / bf16), alone and overlapped on a copy engine"
stage r68.verdict 68 manual  -    -                        r68.cross,r68.balance "spec 17 §9: the GO / NO-GO line by spec 17 §3's rule"
rownote 68 "Starts only after r22.devices and r22.p1 PASS (spec 17's park). Every TP stage is gpu (two cards). The GO also needs Bolt's rows 59-62 passed."
```
- [ ] **Step 2: commit** `git commit -S -m "docs: box queue row 68 - the TP P0 (spec 17a)"`.

**Gate for the plan:** the host tests green; on the box the link, peer, crossing, step and balance tables recorded with
both cards idle, and the verdict line written in spec 17 §9. Hand back with the tables and the line.
