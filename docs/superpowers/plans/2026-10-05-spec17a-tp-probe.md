# Spec 17a - tensor parallel P0, Qwen3.8 first: both all-reduce routes (P2P peer writes, host-staged + own BAR), the per-card step, the go / no-go line

**Status (2026-10-27): PLANNED - Qwen3.8 first (spec 17 §R2); third in the Qwen speed track, next after spec 28
(plans 28a / 28b) and plan 19f, when the operator says go.** Nothing built. Box queue row **71** (the next free row after
67 once spec 28 took 68 / 69 and plan 19f 70; spec 28's header lists the track's rows 68-75). The revision below
supersedes this plan's conditions, goal and tasks for the first P0; the 2026-10-23 text after it (Bolt first) is kept
for Bolt's later P0, which takes the next free row when it is run.

**Status (2026-10-23, superseded 2026-10-27):** PARKED - began only after `r22.devices` / `r22.p1` passed. Rewritten
2026-10-23 for the Bolt-first ruling (spec 17 §R); the 2026-10-05 version is in git history.

## Revision 2026-10-27 - Qwen3.8 first

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax for tracking. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the numbers spec 17 §R2.7's rule needs, and the rule applied, for Qwen3.8 (Agnes derived beside it): the two
cards' link, BAR and P2P facts; how the CPU writes into a card's own VRAM through its BAR; the **all-reduce's per-call
latency at 1 and 8 rows and its per-token cost at 128 calls** on both routes - **(P)** direct P2P peer writes (push and
pull) and **(H)** host-staged + own-BAR (with **(H')**, the cards reading the sum from pinned host memory, as the
fallback); the per-card step at TP shapes; the GO / NO-GO line in spec 17 §9.

**Architecture:** spec 17 §R2.5-§R2.7 (and §4.1's P2P rules, §1's facts). A new probe `tools/probe/probe_tp` (C++ /
Level Zero, one process, one context over both cards - the multi-device context code spec 16b built) with its kernels
in `tools/probe/probe_tp.cl`: the fold-then-send arm (a card folds its split-K slices into one fp32 row, writes it and
a sequence stamp to the route's destination, polls the result's stamp with system-scope atomics, sums or takes the sum,
continues), inside one replayed list per card; for (H) a host thread on a pinned core that sums the two rows in the
fixed order (card 0's, then card 1's) in fp32 and stores the sum and stamp into both cards' VRAM. Shapes from
`ModelDesc` (`tools/probe/tp_shapes.h`, a `tp_rank` / `tp_size` view of spec 17 §R2.5); the per-card GEMV times
through `probe_small_m`'s control arm (plan 28a) at the per-card shapes.

**Tech Stack:** C++17, OpenCL C, Level Zero; the box's custom kernel (7.3.0-rc4-b70-p2p) for the (P) arms.

**Spec:** `docs/superpowers/specs/2026-10-05-spec17-tensor-parallel-design.md` (§R2, §1, §4.1's P2P rules, §5, §9).
Precedents: spec 16b (`pp_handoff.cl`'s system-scope flag protocol, `pp_protocol_test`), plan 22a (`offload_links.sh`,
bounded waits), plan 28a (`probe_small_m`, `probe_verify_m`'s profiled lists).

### Dependencies and branch points

- Main with plans 28a / 28b and 19f merged (the track's order). Branch **`spec17a-tp-probe`** from main.
- **Box:** both cards idle; the first session runs `r22.devices` and `r22.p1` first (two cards in one process; their
  result is recorded beside this row's - a failure there does not stop 17a, whose probe owns its context code, but it
  is named in the verdict). The gate checkpoint (`qwen`) for the step arm.

### Global Constraints

- Signed commits after every step; no merge, no push. Box tree automatic; `tools/box.env` copied if missing, never
  committed. No `rm -rf`. No IP address or host name in any file.
- **Every two-card stage is `gpu`** (the whole box through `tools/gpu_lock.sh box`); host / CPU stages `cpu`. Detached
  and polled. Idle box for every timed arm (`uptime` and the idle grade recorded); interleaved pairs after a warm-up,
  median of 5 with the range.
- No engine code changes: `tools/kernel_cmdlines` +0 / -0 / ~0 for `src/`; G0's sha is the check.
- Every number with its grade (spec 17's legend); published figures (the external two-B70 engine's ~33 µs a call)
  appear only as the bar to beat, never as our number.

### Review Focus

1. **Latency is the number:** an all-reduce of 20 KiB is ~2 µs of transfer at 12 GB/s (derived, assumed rate); the arm
   times the whole call - the last store, the write landing, the CPU or the peer noticing it, the sum, the result
   landing, the card noticing it - inside replayed lists, minus the same lists without calls.
2. **Both routes, push and pull:** (P) push and (P) pull, (H) and (H'); a route that does not work on this platform is
   recorded with its error, not dropped silently.
3. **Bitwise on both cards in every arm**, the fixed order (card 0, then card 1) on every route; stamps detect a stale
   or missing write (an injected skipped write must be caught) - 10^4 replays.
4. **Sizes from `ModelDesc`:** 128 calls (Qwen3.8) / 144 (Agnes), 20 KiB a row, the per-card shapes of §R2.5 - no
   constants in the probe.
5. **The CPU in route (H):** one spinning core per stream; its load and any interference with the host's own work (the
   server thread) recorded.

### Task 1: the probe and its host twin (Mac)

**Files:**
- Create: `tools/probe/probe_tp.cc`, `tools/probe/probe_tp.cl`, `tools/probe/tp_shapes.h`,
  `tests/kernels/tp_shapes_test.cc` (Qwen3.8 128 + 1 calls and Agnes 144 + 1 at 20 KiB a row; §R2.5's per-card shapes,
  each K slice whole 64-groups, each fused layout re-fused per card), `tests/runtime/tp_crossing_protocol_test.cc`
  (beside `pp_protocol_test`: two "card" threads and the CPU-sum thread - stamps, bounded spins, the fixed-order sum
  bitwise, a dropped write detected; both routes' protocols)
- Modify: `tools/probe/CMakeLists.txt`, `tests/CMakeLists.txt`

**Interfaces:** `probe_tp facts|bar|ar|step [--model qwen38|agnes] [--route p2p-push|p2p-pull|host-bar|host-read]
[--rows 1|8] [--calls N] [--rounds N] [--replays N]`; every wait bounded (a timeout prints the arm and exits 1).

- [ ] **Step 1: failing tests** (`tp_shapes_test`, `tp_crossing_protocol_test`).
- [ ] **Step 2: implement, PASS;** `tools/mac_check.sh --base main --quick` green. Commit
  `git commit -S -m "probe: probe_tp - the TP all-reduce arms (P2P push / pull, host-staged + own BAR, host read) and their host twin, Qwen3.8 first (spec 17a)"`.

### Task 2: the facts (box, `r71.facts`)

- [ ] **Step 1:** `offload_links.sh` idle and `--under-load` (both cards' LnkCap / LnkSta and every bridge: replaces
  the operator-stated PCIe 3.0 x16 with a measured line; a STOP line below Gen3 x8 is a NO-GO input); the BAR lines
  (`lspci -vv` region sizes, both cards); the running kernel's version string; `zeDeviceCanAccessPeer` and the P2P
  properties both ways. Record in `docs/probe-tp-<date>.md` §1. Commit
  `git commit -S -m "probe: the two cards' links, BARs and peer access, measured (spec 17a)"`.

### Task 3: the CPU into VRAM through the BAR (box, `r71.bar`)

- [ ] **Step 1:** `probe_tp bar`: each candidate path for a CPU store into a Level Zero device allocation (spec 17
  §R2.6) - which works, its setup, and the CPU-store-to-card-sees-it latency for a 20 KiB / 160 KiB row plus a stamp
  (the card polling with system-scope atomics), median of 5; and the card-to-pinned-host store's latency the other way.
  If no path works, record why and mark (H) as (H') only. Record §2. Commit
  `git commit -S -m "probe: CPU stores into VRAM through the BAR - the path and its latency (spec 17a)"`.

### Task 4: the all-reduce (box, `r71.ar`)

- [ ] **Step 1:** `probe_tp ar` for each route at `--rows 1` and `--rows 8`: the **per-call latency** (a chain of calls,
  each waiting for the previous) and the **per-token cost at 128 calls** (Qwen3.8; 144 for Agnes, derived or measured)
  inside two replayed lists against the same lists without calls; bitwise on both cards over 10^4 replays; the injected
  stale write caught; for (H) the CPU core's load. Interleaved pairs, median of 5, idle. Record §3 beside the published
  bar (~33 µs a call, ~4.2 ms a token). Commit
  `git commit -S -m "probe: the TP all-reduce on both routes at 1 / 8 rows and 128 calls (spec 17a)"`.

### Task 5: the per-card step (box, `r71.step`)

- [ ] **Step 1:** the per-card linears of §R2.5 at M = 1 (and M = 8 through `gemm_sm`, 28b) - `qkv‖z` 5120 x 8192 L1,
  `out_proj` / `o_proj` 3072 x 5120, `q‖k‖v` 5120 x 7168, `gate‖up` 5120 x 17408, `down` 8704 x 5120, the int8 head
  5120 x 124160 - each x its layer count, interleaved against the one-card shapes; the one-card list's other launches
  from `probe_verify_m`'s profile, the per-head ones (attention, `gdn_step`) halved (derived). The derived per-card step
  and, with Task 4's cost, the derived plain TP decode for Qwen3.8 (and Agnes). Record §4. Commit
  `git commit -S -m "probe: the per-card Qwen3.8 step at TP shapes, the derived TP decode (spec 17a)"`.

### Task 6: the verdict

- [ ] **Step 1:** spec 17 §9 "P0 verdict": the tables, the route chosen, and **the line** by spec 17 §R2.7:
  - **GO** when (a) the chosen route is bitwise on both cards over 10^4 replays with stale writes caught, (b) 128 calls
    at M = 1 cost **<= ~4.0 ms a token**, and (c) the derived plain TP decode is **>= ~44 t/s**;
  - **the operator's call** when (b) is 4.0-5.0 ms or (c) is 40-44 t/s;
  - **NO-GO** (TP stays parked for Qwen3.8; the record names the failed condition and the margin) when (a) fails, (b)
    exceeds 5.0 ms, (c) is below 40 t/s, or a link trains below Gen3 x8.
  Recorded beside it: the M = 8 cost, (P) against (H), (H)'s CPU load, Agnes's derived figure. `docs/README.md`: the
  probe record's line. Commit `git commit -S -m "spec 17: TP P0 verdict on Qwen3.8 - <GO|NO-GO|operator's call>"`.

### Task 7: the box queue row

- [ ] **Step 1:** row **71** in `box-validation-queue.md` and its block in `tools/box_validate/stages.sh`
  (`python3 tools/box_validate/test_box_validate.py` passes):

```text
row 71 "spec 17a - the TP P0, Qwen3.8 first: links / BARs / peer access, CPU stores into VRAM through the BAR, the all-reduce on both routes (P2P push / pull, host-staged + own BAR, host read) at 1 / 8 rows and 128 calls, the per-card step at TP shapes, the go / no-go line (plan 17a)"
stage r71.k0      71 default cpu  -     g0.sha,g0.bitwise,g0.suite "G0: no src/ binary changes (the probe's kernels are tools/probe's)"
stage r71.host    71 default cpu  -     -                     "tp_shapes_test, tp_crossing_protocol_test"
stage r71.facts   71 default gpu  -     r22.devices           "offload_links.sh idle and --under-load, the BAR lines, the kernel version, peer access both ways; a STOP line stops the row"
stage r71.bar     71 default gpu  -     r71.facts             "probe_tp bar: the CPU-into-VRAM path and its latency at 20 / 160 KiB; the card-to-pinned-host latency"
stage r71.ar      71 default gpu  -     r71.bar               "probe_tp ar: P2P push / pull, host-bar, host-read at 1 / 8 rows; per call and per token at 128 calls inside replayed lists; bitwise over 10^4 replays, a stale write caught; idle, median of 5"
stage r71.step    71 default gpu  qwen  r71.k0                "the per-card linears at TP shapes (M 1; M 8 through gemm_sm) and the profiled one-card list: the derived per-card step and TP decode"
stage r71.verdict 71 manual  -    -     r71.ar,r71.step       "spec 17 §9: the GO / NO-GO line by spec 17 §R2.7"
rownote 71 "Run r22.devices and r22.p1 first in the same session (two cards in one process); their result is recorded beside this row. Every two-card stage holds the whole box. Hand back: the facts, the BAR path, the all-reduce table (per route, rows, per call, per token, bitwise, stale caught, CPU load), the per-card step table."
```
- [ ] **Step 2: commit** `git commit -S -m "docs: box queue row 71 - the TP P0 on Qwen3.8 (spec 17a)"`.

**Gate for the plan:** the host tests green and `tools/mac_check.sh --base main` PASS (cmdlines ~0 for `src/`); on the box
the facts, BAR, all-reduce and step tables recorded with both cards idle, and the verdict line written in spec 17 §9.
Hand back with the tables and the line.

---

## The 2026-10-23 plan (Bolt first) - superseded for the first P0, kept for Bolt's later P0

Its row (68, proposed on 2026-10-23) is withdrawn: 68 is plan 28a's. Bolt's P0 takes the next free row when it is run;
its arms reuse `probe_tp` (Task 1 above) with Bolt's shapes added to `tp_shapes.h`. LiLiCorr (spec 26) was dropped on
2026-10-27: Bolt has no drafter, so the text's speculation references are void.

**Conditions (operator, 2026-10-23):** this plan **starts only after `r22.devices` and `r22.p1` have PASSed on the box**
(spec 16's first two-card rows: two cards in one process, peer access, the `copy` / `peer` hand-offs bitwise). It is
standalone: it depends on no spec 16a artefact (16a never ran; `docs/probe-multi-gpu-2026-10-05.md` and
`tools/probe/probe_p2p.*` do not exist) and owns its probe.

(The 2026-10-23 status line moved to the top of this file.)

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
