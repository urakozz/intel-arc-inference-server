# Spec 28m - MoE small-M probe: the expert kernels at M = 1..8 on every MoE engine, distinct-expert statistics, the go / no-go line per engine

**Status (2026-11-01): PLANNED, reviewed 2026-11-01 (spec 28m's review corrections; operator decisions in spec 28m §7)
- starts when the operator says go.** Nothing built. **No box queue row** is allocated
(the operator has not asked; next free row 77). The stages below are described in prose; add rows only on request.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. One implementing agent for the whole plan; the gates below are the review (no per-task reviewer). Steps use checkbox (`- [ ]`) syntax. **Commit after every step** (signed, `git commit -S`; a step that does not build or pass yet goes in as a `WIP:` commit) and keep a short handoff file at the worktree root (untracked) when you stop.

**Goal:** the numbers spec 28m §4 needs, per engine, and the rule applied: today's expert kernels against the union
stand-ins (U1 route-sorted, U2 shared-weight work-group, U3 DPAS) at M = 1..8; the distinct-expert statistics D(M) from
real routes; the verify step's attribution for the engines that have one (Ornith, Flash-Next); numerics and row
independence; the verdict in spec 28m §9. **Order (operator 2026-11-01):** Ornith first; K2 only if the operator says so
after Ornith's numbers; then Flash-Next; Bolt and Kolibri as shape rows only.

**Architecture:** spec 28m §3. A new probe `tools/probe/probe_moe_small_m` (C++ / Level Zero, one card) with stand-in
kernels in `tools/probe/probe_moe_small_m.cl` (one source; compile-time engine shape, M, SLOTS, E, TOP_K, INTER, HIDDEN,
shared-expert form, FORM = today | u1 | u2 | u3); today's `moe.cl` / `q4_moe.cl` / `k2_moe.cl` / `kol_moe.cl` /
`nb_moe.cl` kernels as the control (probe-only builds at an M the engine's CMake stops short of); U2 recovered from spec
26c's `nb_moe_union.cl` (`git show 0219cc2^:src/kernels/bolt/nb_moe_union.cl`) and re-targeted per family; a host fp64 reference
`tests/kernels/moe_small_m_ref.h` (expert SwiGLU / relu2, each family's combine order - rank for Ornith / Flash-Next, ascending id for K2 / Kolibri / Bolt) shared with a future build
plan; `probe_verify_m`-style engine ownership to read route buffers (`ProfileEvents*` already exists).

**Tech Stack:** C++17, OpenCL C (Intel sub-group matrix builtins for U3), Level Zero, CMake.

**Spec:** `docs/superpowers/specs/2026-11-01-spec28m-moe-small-m-probe-design.md`. Precedents: plan 28a
(`2026-10-27-spec28a-small-m-probe.md`, the structure, the timing protocol, the ISA check), docs/probes/probe-small-m-gemm-
2026-11-01.md (a record to model §0-§6 on), `tools/oracle/eagle3_cost.py` (the iid formula and the expert-union counter),
docs/probes/probe-eagle3-k2-2026-10-08.md (K2's recorded routes).

## Dependencies and branch points

- Main (ac98878 or later). Branch **`spec28m-probe`** from main. Plans 28b / 28c need not be built: the dense part of
  (b) uses 28a's derived 1.1x GEMM-sum ratio, labelled so, until 28b's measured table exists.
- **Box (later, operator's go):** the gate checkpoints per engine - data keys for Ornith (`urakozz/Ornith-1.5-35B-A3B-W4A16-
  AutoRound-GPTQ`), K2-Horizon, Flash-Next (`--layers`), Kolibri / Bolt only for shapes (synthetic weights are enough for
  shape rows). One card; the GPU lock; DRM-holder check at dispatch. **K2's recorded routes are on the Mac**
  (`oracle-out-eagle3-k2/dumps/*.k2taps.safetensors`, the CPU reference's golden-prompt routes from the EAGLE3 probe);
  copy them to the box only if K2's block timing is wanted on recorded routes.
- No other plan needs to finish first; nothing here edits `src/`.

## Global Constraints

- Signed commits after every step; no merge, no push. Box tree automatic; `tools/box.env` copied if missing, never
  committed. No `rm -rf`. Python probes are scp'd, not heredoc'd.
- **No engine code changes:** `tools/kernel_cmdlines` +0 / -0 / ~0 for `src/` (stand-ins and probe-only builds are
  `tools/probe`'s); G0's sha is the check.
- **Timing:** interleaved control / candidate pairs after a warm-up, median of 5 (kernels) / 3 (steps), 3 % noise line,
  device 0, the GPU lock. **One-card busy-box timing is allowed, labelled "busy-box timing (one card), to be re-checked on
  a quiet box"** (operator 2026-11-01), with the box load printed beside every table and a re-check list in the record.
  Detached and polled (`tools/probe/detach.sh`).
- Every number carries its grade (spec 28's legend); estimates show their formula.
- No new CLI flag, no environment variable for selection (the probe's own CLI is a tool, not production).
- 256-GRF builds carry the ocloc option; the probe reads the built binary's GRF mode before a timed arm.
- No host-RAM offload on any timed path.

## Review Focus

1. **Like for like:** control and stand-in read the same expert bytes (the engine's layout-1 blocks, same addresses via the
   engine's expert-address macro), the same activations and routes, and write the same outputs (the residual fold
   included), so a ratio is the union form's effect and nothing else.
2. **Static command list:** the union forms use a fixed grid of M x k slots with early exit; the probe asserts the launch
   count does not depend on the routes.
3. **Real routes, not only synthetic:** the verdict's (a) is on recorded routes; iid and the all-distinct / all-equal
   bounds are recorded beside, never instead.
4. **Per-row arithmetic unchanged for U1 / U2:** bitwise against today's kernel on every route pattern, at every M.
5. **The whole step:** (b) is the real list's non-expert time plus (a)'s block; profiled shares are applied to the
   unprofiled step (the flush per launch inflates absolutes).

---

### Task 1: the probe, the stand-ins and the host reference (Mac)

**Files:**
- Create: `tools/probe/probe_moe_small_m.cc`, `tools/probe/probe_moe_small_m.cl`, `tests/kernels/moe_small_m_ref.h`,
  `tests/kernels/moe_small_m_ref_test.cc`
- Modify: `tools/probe/CMakeLists.txt`, `tests/CMakeLists.txt`, `tools/mac/opencl/intel_shim.h` only for a missing
  declaration (never an emulation)

**Interface:** `probe_moe_small_m routes|block|attrib|numerics [--engine ornith|k2|flashnext|kolibri|bolt] [--shape ...]
[--m 1,2,3,4,8] [--form today|u1|u2|u3] [--routes recorded|iid|distinct|same] [--rounds 5] [--device 0]`.

- [ ] **Step 1: failing tests** - `moe_small_m_ref_test`: the reference against each engine's host twin (the repo's existing
  MoE host tests: Ornith's, K2's `k2_moe` host test, Flash-Next's, Kolibri's, Bolt's) within 1 fp32 ulp of the sum;
  the route-pattern generators (iid, all-distinct, all-equal, K2-locality resampler calibrated so D(2..4) = 12.46 / 15.75 /
  18.55 at E 100, k 8); the iid formula against `eagle3_cost.py`'s union counter.
- [ ] **Step 2: implement, PASS.** U1 (gate||up through a permutation table sorted by expert id; down unchanged), U2
  (`nb_moe_union.cl`'s schedule per family: one work-group set per distinct expert, every pair of it in (row, slot) order,
  sets past the union exit; down writes per-pair bf16 d[row][slot][n] and a third launch combines each row in the
  family's order - rank order + gated shared + residual fold for Ornith / Flash-Next, ascending id for K2 / Kolibri /
  Bolt; each family file's `FP_CONTRACT` regime kept), U3 (DPAS rows-on-A). Kernels pass `tools/mac/cl_syntax.sh`; `tools/box_check.sh --base main` green (the Mac has
  no DPAS, so U3 and all numerics run on the box). Commit
  `git commit -S -m "probe: probe_moe_small_m - union stand-ins (route-sorted, shared-weight, DPAS) for the MoE expert kernels and their fp64 reference (spec 28m)"`.

### Task 2: routing statistics (box, Ornith and Flash-Next; Mac replay for K2)

- [ ] **Step 1:** `probe_moe_small_m routes`: with an engine owned in-process, run the golden prompts and the A4 tool-call
  set through the **verify lists at M = 2..4** (Ornith, Flash-Next; their MTP drafts as the rows) and read the route buffers
  after each step: D(M) per layer (min / median / max over layers), the ratio to the iid prior; also D(M) over accepted
  rows only, the like-for-like with K2's record. For K2 (no verify list, Mac): replay the recorded CPU-reference routes
  from the EAGLE3 probe as M consecutive rows - this reproduces 12.46 / 15.75 / 18.55 as the check of the replay itself
  (golden prompts only: the A4 set was never dumped for K2; label it so). For Kolibri and Bolt: nothing to record (no list, no checkpoint run); their rows use the
  K2-locality resampler and iid. Record spec-28m-record `docs/probes/probe-moe-small-m-<date>.md` §1. Commit
  `git commit -S -m "probe: distinct experts per M rows on Ornith and Flash-Next, K2 replay (spec 28m)"`.

### Task 3: the expert block (box)

- [ ] **Step 1:** for each engine's shapes (Ornith; K2 experts and MoVA; Flash-Next; Kolibri; Bolt's 12 (E, I) pairs):
  the expert block at M = 1..4 (to 8 for K2 lookup, Ornith / Flash-Next record only), today against U1 / U2 (/ U3), on
  recorded routes where Task 2 has them, iid, all-distinct, all-equal; ms per block, GB/s against **distinct-expert bytes**
  and against M x k bytes, the ratio to today's M = 1 block. ISA check (Review Focus: dequant count must not grow with
  rows in U2; `ocloc -dump` at M = 2 / 4 / 8; and whether `tile_dot`'s `acc += gacc * scale` compiles to a mad in today's
  `moe.cl` / `k2_moe.cl` binaries and in U2's - they must match for U2's bitwise claim). Record §2. Commit
  `git commit -S -m "probe: the MoE expert block at M = 1..8, today's kernels against union forms (spec 28m)"`.

### Task 4: attribution (box, Ornith and Flash-Next)

- [ ] **Step 1:** `probe_moe_small_m attrib`: the real verify list at M = 1..4 at depth 4096 (and 32768), the unprofiled step
  (interleaved against M = 1) and the profiled per-family table (dense GEMV, experts, attention, GDN, head, other). For
  Ornith reuse the existing tools instead of new code where they fit: `probe_mtp_steps <ornith> 4096 32 3 int8` (stage
  r16.cost, postponed - run it here) for the unprofiled verify, `probe_verify_m --verify8 1`'s profiled mode capped at
  M = 4 with an experts family for `moe_*` names (two small probe edits). Flash-Next
  on its `--layers` truncation, labelled so (its step ratio is of the truncated model). Record §3. Commit
  `git commit -S -m "probe: the verify step at M = 1..4 on Ornith and Flash-Next, experts against the rest (spec 28m)"`.

### Task 5: numerics (box)

- [ ] **Step 1:** U1 / U2 bitwise against today's kernel on every route pattern at M = 2..8; U3 against the fp64 reference
  (N2's bound) and against today's; property R-moe (10^3 trials per shape: rows bitwise across M >= 2 and with the other
  rows' routes randomised); the router, `gemv_bf16`, untouched (checksum). Record §4. Commit
  `git commit -S -m "probe: MoE small-M numerics and row independence (spec 28m)"`.

### Task 6: the verdicts

- [ ] **Step 1:** spec 28m §9, per engine, by §4's rule: (a) from Task 3 on recorded routes, (b) derived verify(4) =
  Task 4's non-expert time + (a)'s block (dense at 28a's 1.1x until 28b is measured), (c) Task 5. Engines without a list:
  (a), (c) and the bytes-model (b), labelled derived. State per engine GO / operator's call / NO-GO, the form (U1 / U2 / U3),
  and the next lever on NO-GO. `docs/README.md`: the record's line. Commit
  `git commit -S -m "spec 28m: MoE small-M verdicts - <per engine>"`.

### Stages in prose (no queue rows)

1. **Host (Mac):** Task 1's tests and `mac_check`.
2. **Routing (box, one card):** Task 2, Ornith (Flash-Next after), golden + A4 prompts; short. K2's replay on the Mac.
3. **Block timing (box):** Task 3 - Ornith, then the Bolt / Kolibri shape rows; K2 only on the operator's word after
   Ornith's numbers; Flash-Next last (weights synthetic where no checkpoint is on the card).
4. **Attribution (box):** Task 4, Ornith then Flash-Next.
5. **Numerics (box):** Task 5.
6. **Verdict (manual).**
Timed stages need the idle protocol unless the operator allows busy-box labelled timing again; the record says which.

**Gate for the plan:** `moe_small_m_ref_test` green and `tools/box_check.sh --base main` PASS on the Mac (cmdlines ~0 for
`src/`); on the box the routing, block, attribution and numerics tables recorded, and the verdict lines in spec 28m §9.
Hand back with the tables and the lines.
